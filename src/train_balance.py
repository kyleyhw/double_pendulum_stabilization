r"""
Phase S: SAC that balances — reverse curriculum from the LQR basin to hanging.

Why this trainer exists
=======================
Every earlier RL phase started episodes from hanging and asked exploration to
find the up-up capture set, an event of (practically) measure zero on the
chaotic double-pendulum energy surface (``docs/NEXT_STEPS.md``). This trainer
inverts the problem (Florensa et al. 2017, *Reverse curriculum generation*):

1. **Basin stage.** Episodes start *inside* the upright basin — pole angles
   uniform in :math:`\pm r`, angular rates in :math:`\pm 2r`, cart position and
   velocity in :math:`\pm 0.5` — and :math:`r` grows from 0.05 to 0.25 rad as
   the deterministic policy masters each radius.
2. **Trajectory stage.** Episodes start on the Phase R swing-up trajectory
   (``src/control/data/swingup_d1.npz``) at a time :math:`\tau` drawn
   uniformly from :math:`[\tau_{\min}, T]`, plus the environment's own
   :math:`\pm 0.05` reset noise on every state component; 25 % of resets stay
   in the basin. :math:`\tau_{\min}` moves backwards from :math:`T` to 0 (=
   hanging at rest) as the frontier is mastered.

The trajectory is used **only for initial states** — never as an action
target — so what is learned is a state-feedback policy that swings up and
balances on its own.

Design choices that differ from ``src/train_sac.py`` (and why)
-------------------------------------------------------------
* ``ForceControl`` with :math:`F_{\max} = 100` N, not ``VelocityControl``
  (whose :math:`10^5` N per unit action made balance-scale corrections
  invisible to exploration). The Phase R swing-up peaks at 27 N.
* Action repeat 4 (control at 50 Hz) so :math:`\gamma = 0.99` spans 2 s.
* Fixed observation scaling (no running statistics) — the reset distribution
  changes over the curriculum and a drifting normaliser would shift inputs
  under the critic.
* Physics are fixed at full difficulty :math:`\delta = 1` (:math:`g = 9.81`,
  frictionless, wind :math:`\sigma_w = 1` N per simulator step): the
  curriculum is over *initial states*, not physics.
* Reward (per control step, bounded in :math:`[-0.1, 1]`):

  .. math::

      r = \tfrac12 e^{-(e_1^2 + e_2^2)/(2\cdot 0.25^2)}
        + \tfrac12 \cdot \tfrac{(1+\cos e_1)(1+\cos e_2)}{4}
        - 0.05\,(x / x_{\rm soft})^2
        - 0.002\,\min(\dot\theta_1^2 + \dot\theta_2^2, 50)

  where :math:`e_i` is pole *i*'s angle from upright. Leaving
  :math:`|x| \le x_{\rm soft}` terminates the episode (no further reward).

Goal-conditioned mode (``--goals all``, Phase 6)
------------------------------------------------
The same recipe for all four equilibria (DD, UU, DU, UD —
:mod:`src.control.switching`). The observation gains a one-hot goal, the
reward's :math:`e_i` are measured from the goal's pole angles, the basin
stage is run around every goal, and the trajectory stage starts episodes on
the switching library's 12 transitions at a time-to-go :math:`h` before the
goal. The curriculum advances when the frontier success is at least
``--advance_at`` overall and no goal is more than 0.15 below it.

Training uses the vectorised plant in :mod:`src.control.swingup` (verified
equal to the environment's RK4 to :math:`10^{-12}`); evaluation
(``tools/eval_balance.py``) runs the real ``DoublePendulumCartEnv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.sac import SACAgent  # noqa: E402
from src.control.swingup import PlantParams, SwingUpTrajectory, rk4_step, wrap_angle  # noqa: E402
from src.control.switching import ANGLES, DEFAULT_LIBRARY, NAMES, load_library  # noqa: E402

DEFAULT_TRAJ = os.path.join(os.path.dirname(__file__), "control", "data", "swingup_d1.npz")
OBS_SCALE = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0 / 3.0, 1.0 / 8.0, 1.0 / 8.0])
STRICT = 0.17
GOAL_ANGLES = np.array([ANGLES[n] for n in NAMES])  # (4, 2), rows in NAMES order


def make_obs(states: np.ndarray, goals: np.ndarray | None = None) -> np.ndarray:
    """Scaled observation ``[x, sin θ1, sin θ2, cos θ1, cos θ2, ẋ, θ̇1, θ̇2]`` (env
    layout), plus a one-hot goal over :data:`NAMES` when ``goals`` (indices) is given."""
    s = np.atleast_2d(states)
    raw = np.concatenate([s[:, :1], np.sin(s[:, 1:3]), np.cos(s[:, 1:3]), s[:, 3:6]], axis=1)
    obs = (raw * OBS_SCALE).astype(np.float32)
    if goals is None:
        return obs
    onehot = np.zeros((len(s), len(NAMES)), dtype=np.float32)
    onehot[np.arange(len(s)), np.atleast_1d(goals)] = 1.0
    return np.concatenate([obs, onehot], axis=1)


def balance_reward(states: np.ndarray, x_soft: float, goals: np.ndarray | None = None
                   ) -> np.ndarray:
    """Per-step reward (module docstring) relative to each env's goal (default up-up)."""
    target = GOAL_ANGLES[1] if goals is None else GOAL_ANGLES[np.atleast_1d(goals)]
    e = wrap_angle(states[:, 1:3] - target)
    gauss = np.exp(-(e ** 2).sum(axis=1) / (2.0 * 0.25 ** 2))
    height = (1.0 + np.cos(e[:, 0])) * (1.0 + np.cos(e[:, 1])) / 4.0
    x_pen = 0.05 * (states[:, 0] / x_soft) ** 2
    v_pen = 0.002 * np.minimum((states[:, 4:6] ** 2).sum(axis=1), 50.0)
    return 0.5 * gauss + 0.5 * height - x_pen - v_pen


class InitSampler:
    r"""
    Reverse-curriculum initial-state distribution (module docstring).

    ``trajs`` maps a goal index to the trajectories that end at that goal. The
    curriculum state is the basin radius ``r`` and the *reach* :math:`h`: the
    time-to-go before the end of a trajectory from which episodes may start
    (:math:`h = 0` is the goal itself; :math:`h = T` the trajectory's start).
    Training starts are drawn with :math:`h` uniform in :math:`[0, h_{\rm cur}]`;
    the frontier is :math:`h = h_{\rm cur}`.
    """

    def __init__(self, trajs: dict[int, list[SwingUpTrajectory]], rng: np.random.Generator,
                 r_start: float = 0.05, r_max: float = 0.25, r_step: float = 0.05,
                 tau_step: float = 0.1, basin_frac: float = 0.25) -> None:
        self.trajs = trajs
        self.goals = sorted(trajs)
        self.rng = rng
        self.r = r_start
        self.r_max = r_max
        self.r_step = r_step
        self.reach = 0.0
        self.reach_max = max(t.duration for ts in trajs.values() for t in ts)
        self.tau_step = tau_step
        self.basin_frac = basin_frac

    @property
    def stage(self) -> str:
        return "basin" if self.reach <= 0.0 else "trajectory"

    @property
    def done(self) -> bool:
        return self.reach >= self.reach_max - 1e-9

    def basin(self, goals: np.ndarray, r: float | None = None) -> np.ndarray:
        r = self.r if r is None else r
        n = len(goals)
        s = np.zeros((n, 6))
        s[:, 0] = self.rng.uniform(-0.5, 0.5, n)
        s[:, 1:3] = GOAL_ANGLES[goals] + self.rng.uniform(-r, r, (n, 2))
        s[:, 3] = self.rng.uniform(-0.5, 0.5, n)
        s[:, 4:6] = self.rng.uniform(-2 * r, 2 * r, (n, 2))
        return s

    def on_trajectory(self, goals: np.ndarray, reach: np.ndarray) -> np.ndarray:
        s = np.empty((len(goals), 6))
        for i, (g, h) in enumerate(zip(goals, reach, strict=True)):
            ts = self.trajs[int(g)]
            t = ts[self.rng.integers(len(ts))]
            j = int(round(max(t.duration - h, 0.0) / t.params.dt))
            s[i] = t.states[j]
        return s + self.rng.uniform(-0.05, 0.05, s.shape)

    def _goals(self, n: int) -> np.ndarray:
        return self.rng.choice(self.goals, size=n)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        goals = self._goals(n)
        if self.reach <= 0.0:
            return self.basin(goals), goals
        s = self.on_trajectory(goals, self.rng.uniform(0.0, self.reach, n))
        use_basin = self.rng.random(n) < self.basin_frac
        s[use_basin] = self.basin(goals[use_basin], self.r_max)
        return s, goals

    def frontier(self, n: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Frontier starts (equal numbers per goal), their goals, and the time needed
        to finish from there."""
        goals = np.resize(np.array(self.goals), n)
        if self.reach <= 0.0:
            return self.basin(goals), goals, 0.0
        return self.on_trajectory(goals, np.full(n, self.reach)), goals, self.reach

    def advance(self) -> str:
        if self.reach <= 0.0 and self.r < self.r_max - 1e-9:
            self.r = min(self.r + self.r_step, self.r_max)
            return f"basin radius -> {self.r:.2f} rad"
        self.reach = min(self.reach_max, self.reach + self.tau_step)
        return f"trajectory reach -> {self.reach:.2f} s"

    def state_dict(self) -> dict:
        return {"r": self.r, "reach": self.reach, "reach_max": self.reach_max}

    def load_state_dict(self, d: dict) -> None:
        self.r = d.get("r", self.r)
        if "reach" in d:
            self.reach = d["reach"]
        elif "tau_min" in d:  # checkpoints from the single-goal trainer
            self.reach = self.reach_max - d["tau_min"]


class BatchedBalanceSim:
    """``n`` copies of the plant at fixed physics, stepped with action repeat."""

    def __init__(self, p: PlantParams, n: int, frame_skip: int, max_force: float,
                 wind_std: float, x_soft: float, rng: np.random.Generator) -> None:
        self.p, self.n, self.k = p, n, frame_skip
        self.max_force, self.wind_std, self.x_soft = max_force, wind_std, x_soft
        self.rng = rng
        self.states = np.zeros((n, 6))

    def step(self, actions: np.ndarray, states: np.ndarray | None = None) -> np.ndarray:
        s = self.states if states is None else states
        f = np.clip(actions[:, 0], -1.0, 1.0) * self.max_force
        for _ in range(self.k):
            wind = self.rng.normal(0.0, self.wind_std, len(s)) if self.wind_std > 0 else 0.0
            s = rk4_step(s, f + wind, self.p)
        if states is None:
            self.states = s
        return s


def evaluate_frontier(agent: SACAgent, sim: BatchedBalanceSim, sampler: InitSampler,
                      n: int, multi_goal: bool, balance_time: float = 3.0
                      ) -> tuple[float, float]:
    """Deterministic rollouts from the frontier; returns the overall and the worst
    per-goal fraction that end strictly at the goal for the whole final second and
    never leave the soft cart bound."""
    s, goals, reach = sampler.frontier(n)
    ctrl_dt = sim.k * sim.p.dt
    steps = int(round((reach + balance_time) / ctrl_dt))
    last = int(round(1.0 / ctrl_dt))
    ok = np.ones(n, dtype=bool)
    target = GOAL_ANGLES[goals]
    for j in range(steps):
        a = agent.select_action(make_obs(s, goals if multi_goal else None), deterministic=True)
        s = sim.step(np.atleast_2d(a), states=s)
        ok &= np.abs(s[:, 0]) <= sim.x_soft
        if j >= steps - last:
            ok &= np.all(np.abs(wrap_angle(s[:, 1:3] - target)) < STRICT, axis=1)
    per_goal = [ok[goals == g].mean() for g in np.unique(goals)]
    return float(ok.mean()), float(min(per_goal))


def load_trajectories(args: argparse.Namespace) -> dict[int, list[SwingUpTrajectory]]:
    if args.goals == "UU":
        return {NAMES.index("UU"): [SwingUpTrajectory.load(args.traj)]}
    lib = load_library(args.library)
    return {NAMES.index(g): [t for (a, b), t in lib.items() if b == g] for g in NAMES}


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    multi = args.goals != "UU"
    trajs = load_trajectories(args)
    p = next(iter(trajs.values()))[0].params
    sampler = InitSampler(trajs, rng, tau_step=args.tau_step)
    sim = BatchedBalanceSim(p, args.n_envs, args.frame_skip, args.max_force, args.wind_std,
                            args.x_soft, rng)
    eval_sim = BatchedBalanceSim(p, args.eval_episodes, args.frame_skip, args.max_force,
                                 args.wind_std, args.x_soft, np.random.default_rng(args.seed + 1))
    obs_dim = 8 + (len(NAMES) if multi else 0)
    agent = SACAgent(state_dim=obs_dim, action_dim=1, hidden_dim=args.hidden_dim,
                     gamma=args.gamma, lr=args.lr, batch_size=args.batch_size,
                     replay_capacity=args.replay_capacity, alpha_max=args.alpha_max,
                     device=torch.device("cpu"))
    if args.load:
        payload = agent.load(args.load)
        sampler.load_state_dict(payload.get("curriculum", {}))

    os.makedirs(args.log_dir, exist_ok=True)
    run = args.run_name or f"balance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_path = os.path.join(args.log_dir, f"training_log_{run}.csv")
    csv_f = open(csv_path, "w", newline="")  # noqa: SIM115 - closed after the loop
    log = csv.writer(csv_f)
    log.writerow(["transitions", "updates", "episodes", "ep_return", "stage", "r", "reach",
                  "frontier_success", "worst_goal_success", "alpha", "critic_loss", "wall_s"])
    meta = {"max_force": args.max_force, "frame_skip": args.frame_skip,
            "obs_scale": OBS_SCALE.tolist(), "hidden_dim": args.hidden_dim,
            "goals": [NAMES[g] for g in sampler.goals], "multi_goal": multi}

    def save(tag: str) -> str:
        path = os.path.join(args.log_dir, f"{run}_{tag}.pth")
        agent.save(path, extra={"curriculum": sampler.state_dict(), "meta": meta})
        return path

    def goal_arg(g: np.ndarray) -> np.ndarray | None:
        return g if multi else None

    sim.states, goals = sampler.sample(args.n_envs)
    ep_len = np.zeros(args.n_envs, dtype=int)
    ep_ret = np.zeros(args.n_envs)
    returns: list[float] = []
    transitions = updates = episodes = 0
    milestones: dict[str, int] = {}
    diag: dict = {}
    t0 = time.time()
    print(f"[run] {run}  goals={meta['goals']}  F_max={args.max_force} N  "
          f"control dt={args.frame_skip * p.dt:.3f} s  reach_max={sampler.reach_max:.2f} s",
          flush=True)

    while transitions < args.total_transitions:
        obs = make_obs(sim.states, goal_arg(goals))
        if transitions < args.warmup:
            actions = rng.uniform(-1, 1, (args.n_envs, 1)).astype(np.float32)
        else:
            actions = np.atleast_2d(agent.select_action(obs)).astype(np.float32)
        nxt = sim.step(actions)
        rew = balance_reward(nxt, args.x_soft, goals)
        term = np.abs(nxt[:, 0]) > args.x_soft
        agent.buffer.push_batch(obs, actions, rew.astype(np.float32),
                                make_obs(nxt, goal_arg(goals)), term.astype(np.float32))
        ep_ret += rew
        ep_len += 1
        transitions += args.n_envs
        end = term | (ep_len >= args.episode_len)
        if end.any():
            returns.extend(ep_ret[end].tolist())
            episodes += int(end.sum())
            sim.states[end], goals[end] = sampler.sample(int(end.sum()))
            ep_ret[end] = 0.0
            ep_len[end] = 0

        if transitions >= args.warmup:
            for _ in range(int(args.utd * args.n_envs)):
                diag = agent.update() or diag
                updates += 1

        if transitions % args.eval_every < args.n_envs and transitions >= args.warmup:
            success, worst = evaluate_frontier(agent, eval_sim, sampler, args.eval_episodes, multi)
            before = (sampler.r, sampler.reach)
            passed = success >= args.advance_at and worst >= args.advance_at - 0.15
            msg = ""
            if passed and not sampler.done:
                if (sampler.reach <= 0.0 and sampler.r >= sampler.r_max - 1e-9
                        and "basin" not in milestones):
                    milestones["basin"] = transitions
                    save("basin")
                msg = "  ADVANCE: " + sampler.advance()
            elif passed and sampler.done and "full" not in milestones:
                milestones["full"] = transitions
                save("full")
                msg = "  MASTERED full reach"
            recent = float(np.mean(returns[-50:])) if returns else float("nan")
            print(f"[{transitions:8d}] upd={updates} ep={episodes} ret={recent:7.2f} "
                  f"r={before[0]:.2f} reach={before[1]:.2f} frontier_success={success:.2f} "
                  f"worst_goal={worst:.2f} alpha={diag.get('alpha', 0):.3f} "
                  f"({time.time() - t0:.0f}s){msg}", flush=True)
            log.writerow([transitions, updates, episodes, f"{recent:.3f}", sampler.stage,
                          f"{before[0]:.3f}", f"{before[1]:.3f}", f"{success:.3f}",
                          f"{worst:.3f}", f"{diag.get('alpha', 0):.4f}",
                          f"{diag.get('critic_loss', 0):.4f}", f"{time.time() - t0:.0f}"])
            csv_f.flush()
            if transitions % args.save_every < args.n_envs:
                save("latest")
            if "full" in milestones and args.stop_when_mastered:
                break

    final = save("final")
    csv_f.close()
    with open(os.path.join(args.log_dir, f"{run}_summary.json"), "w") as f:
        json.dump({"milestones": milestones, "curriculum": sampler.state_dict(),
                   "transitions": transitions, "updates": updates,
                   "wall_s": time.time() - t0, "args": vars(args)}, f, indent=2)
    print(f"[done] {final}  milestones={milestones}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--goals", default="UU", choices=["UU", "all"],
                    help="UU: Phase S (swing-up + balance). all: goal-conditioned switching "
                         "between the four equilibria (Phase 6) along the switching library.")
    ap.add_argument("--traj", default=DEFAULT_TRAJ)
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
    ap.add_argument("--max_force", type=float, default=100.0)
    ap.add_argument("--frame_skip", type=int, default=4)
    ap.add_argument("--wind_std", type=float, default=1.0)
    ap.add_argument("--x_soft", type=float, default=3.5)
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--episode_len", type=int, default=250, help="control steps")
    ap.add_argument("--total_transitions", type=int, default=3_000_000)
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--utd", type=float, default=0.5, help="gradient updates per transition")
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--replay_capacity", type=int, default=1_000_000)
    ap.add_argument("--alpha_max", type=float, default=0.3)
    ap.add_argument("--tau_step", type=float, default=0.1)
    ap.add_argument("--advance_at", type=float, default=0.9)
    ap.add_argument("--eval_every", type=int, default=20_000)
    ap.add_argument("--eval_episodes", type=int, default=32)
    ap.add_argument("--save_every", type=int, default=200_000)
    ap.add_argument("--stop_when_mastered", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--load", default=None)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
