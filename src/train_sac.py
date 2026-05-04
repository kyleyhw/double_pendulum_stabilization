r"""
SAC training loop for the cart-pendulum task.

Mirrors the high-level structure of ``src/train.py`` (vectorised env
collection, ratchet curriculum, CSV log, best-fallback) but uses SAC's
off-policy update pattern: each env step pushes a transition into the
replay buffer; once warmup is over, one or more gradient updates fire
per env step.

The PPO-specific machinery (GAE, K-epoch minibatches, ratio clipping,
LR / ent / clip schedules, target_kl) is replaced with SAC's:

* automatic entropy tuning (target entropy = -|action_dim|),
* twin-Q + Polyak-averaged target nets,
* state-dependent log_std (the architecture that broke under PPO
  Phase J — works natively under SAC's reparam gradient).

Compute budget: at the c6 pipeline rate (~1.34 s/PPO update at
``--n_envs 8 --rollout_steps 512``), one SAC step (8 env steps + 8
gradient updates) is comparable per env-step but does many more
gradient updates per "PPO update equivalent". Plan around env-steps,
not updates: a 10M env-step SAC run ≈ 5 hours wall on this pipeline.

Curriculum integration
======================
The ratchet uses the same gate as ``train.py``:

.. math::
    \text{advance if }\
    \overline{\text{time\_above}} > 0.9 \delta
    \;\wedge\; \overline{R} > \overline{R}_{\rm prev best}.

The Phase M campaign showed this gate is the *binding constraint* on
curriculum advance once strict-success enters the 4-7 % regime. SAC's
state-dependent variance is the lever expected to break that ceiling
by enabling micro-corrections near upright (raising time_above).

Best-fallback semantics carry over: a deterministic eval every K env
steps; the snapshot from the highest-eval-reward checkpoint is saved
to ``logs/sac_<run>_best.pth``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.sac import SACAgent  # noqa: E402
from src.env.cart_pendulum_base import BatchedEnvRunner  # noqa: E402
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.env.single_pendulum import SinglePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl, VelocityControl  # noqa: E402
from src.strategies.rewards import (  # noqa: E402
    DoublePendulumStandardReward,
    EnergyShapingReward,
    ExponentialSwingUpReward,
    HybridLQRSwingUpReward,
    LQRCostReward,
    RewardStrategy,
    SinglePendulumStandardReward,
)
from src.utils.normalize import RunningMeanStd  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


# ---------------------------------------------------------------------------- #
# Factories — match train.py exactly so configs port over.
# ---------------------------------------------------------------------------- #

def _build_reward(args: argparse.Namespace) -> RewardStrategy:
    curve = getattr(args, "threshold_curve", "linear")
    if args.reward == "exponential":
        return ExponentialSwingUpReward(threshold_curve=curve)
    if args.reward == "standard":
        return (SinglePendulumStandardReward(survival_alpha=args.survival_alpha,
                                             threshold_curve=curve)
                if args.env == "single" else
                DoublePendulumStandardReward(survival_alpha=args.survival_alpha,
                                             threshold_curve=curve))
    if args.reward == "energy":
        return EnergyShapingReward(threshold_curve=curve)
    if args.reward == "lqr":
        return LQRCostReward()
    if args.reward == "hybrid":
        return HybridLQRSwingUpReward(threshold_curve=curve)
    raise ValueError(f"Unknown reward: {args.reward}")


def _build_env(args: argparse.Namespace, seed: int) -> Any:
    control = ForceControl() if args.control == "force" else VelocityControl()
    reward = _build_reward(args)
    cls = SinglePendulumCartEnv if args.env == "single" else DoublePendulumCartEnv
    env = cls(
        reset_mode=args.reset_mode,
        control_strategy=control,
        reward_strategy=reward,
        integrator=args.integrator,
        x_soft=args.x_soft,
        x_hard=args.x_hard,
        boundary_penalty_k=args.boundary_penalty_k,
        wind_max=args.wind_max,
    )
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def _normalize_obs(obs: np.ndarray, rms: RunningMeanStd, *, eps: float = 1e-8,
                   clip: float = 10.0) -> np.ndarray:
    """Apply running-mean/var normalisation to a batch ``(N, obs_dim)``."""
    out = (obs.astype(np.float64) - rms.mean) / np.sqrt(rms.var + eps)
    return np.clip(out, -clip, clip).astype(np.float32)


def _angles_above_threshold(states: np.ndarray, n_poles: int, threshold: float
                            ) -> np.ndarray:
    """Vectorised time-above-threshold check across N envs."""
    angles = states[:, 1:1 + n_poles]
    err = np.abs(np.arctan2(np.sin(angles - np.pi), np.cos(angles - np.pi)))
    return np.all(err < threshold, axis=1)


# ---------------------------------------------------------------------------- #
# Main loop
# ---------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    seed = int(args.seed) if args.seed is not None else int(np.random.randint(0, 100_000))
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"[seed]   {seed}")
    print(f"[config] env={args.env} control={args.control} reward={args.reward} "
          f"n_envs={args.n_envs} algo=sac")

    # Vectorised envs — reuse the BatchedEnvRunner from cart_pendulum_base.
    envs = [_build_env(args, seed=seed + i) for i in range(args.n_envs)]
    runner = BatchedEnvRunner(envs)
    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    n_poles = envs[0].n_poles

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    def _reset_all(seed_offset: int = 0) -> np.ndarray:
        """Reset each env and stack the resulting obs into ``(N, obs_dim)``."""
        obs_list = [envs[i].reset(seed=seed + seed_offset + i)[0] for i in range(args.n_envs)]
        return np.stack(obs_list, axis=0).astype(np.float32)

    def _set_curriculum_all(d: float) -> None:
        for env in envs:
            env.set_curriculum(d)
        runner.sync_curriculum()

    obs_arr = _reset_all()

    agent = SACAgent(
        state_dim=obs_dim, action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma, tau=args.tau,
        lr=args.lr, batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
    )
    print(f"[agent]  device={agent.device} hidden={args.hidden_dim} "
          f"batch={args.batch_size} buffer={args.replay_capacity}")
    if args.load:
        print(f"[load]   {args.load}")
        payload = agent.load(args.load)
        if "obs_rms" in payload:
            obs_rms.load_state_dict(payload["obs_rms"])

    # Logging.
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_{args.env}_{args.control}_{args.reward}_{timestamp}"
    print(f"[run]    {run_name}")
    csv_path = os.path.join(log_dir, f"training_log_{run_name}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "EnvSteps", "Updates", "Episodes", "Reward", "Length", "Difficulty",
        "G", "Friction", "Threshold_Deg",
        "CriticLoss", "ActorLoss", "AlphaLoss", "Alpha", "LogProbMean",
    ])
    tb_writer = SummaryWriter(os.path.join(log_dir, "tb", run_name)) \
        if SummaryWriter is not None else None

    # Curriculum.
    difficulty = float(args.start_difficulty)
    _set_curriculum_all(difficulty)
    best_avg_reward = -float("inf")
    reward_window: collections.deque = collections.deque(maxlen=args.window)
    time_above_window: collections.deque = collections.deque(maxlen=args.window)

    ep_rewards = np.zeros(args.n_envs, dtype=np.float64)
    ep_steps_above = np.zeros(args.n_envs, dtype=np.int64)
    ep_lengths = np.zeros(args.n_envs, dtype=np.int64)
    total_episodes = 0
    total_env_steps = 0
    total_updates = 0
    last_diag: dict[str, float] = {}

    # Best-fallback state.
    best_eval_reward = -float("inf")
    best_eval_snapshot: dict | None = None
    best_eval_obs_rms: dict | None = None
    consecutive_eval_drops = 0

    start_time = time.perf_counter()
    print(f"[warmup] random actions for {args.warmup_steps} env steps")

    try:
        while total_env_steps < args.total_env_steps:
            # Update RMS on raw obs and produce the normalised view for the policy.
            obs_rms.update(obs_arr)
            obs_norm = _normalize_obs(obs_arr, obs_rms)

            if total_env_steps < args.warmup_steps:
                actions = np.random.uniform(-1.0, 1.0,
                                            size=(args.n_envs, action_dim)).astype(np.float32)
            else:
                actions = agent.select_action(obs_norm).astype(np.float32)

            next_obs, rewards, terminated, truncated, _ = runner.step_batch(actions)
            dones = (terminated | truncated)
            # Push transitions to replay buffer (in normalised obs space).
            next_obs_norm = _normalize_obs(next_obs, obs_rms)
            agent.buffer.push_batch(
                obs_norm.astype(np.float32),
                actions,
                rewards.astype(np.float32),
                next_obs_norm.astype(np.float32),
                dones.astype(np.float32),
            )

            # Episode bookkeeping.
            ep_rewards += rewards
            ep_lengths += 1
            raw_states = np.stack([e.state for e in envs], axis=0)
            above = _angles_above_threshold(raw_states, n_poles,
                                            envs[0].reward_strategy.reward_threshold)
            ep_steps_above += above.astype(np.int64)

            for i in range(args.n_envs):
                if dones[i] or ep_lengths[i] >= args.episode_steps:
                    reward_window.append(float(ep_rewards[i]))
                    time_above_window.append(float(ep_steps_above[i]) / args.episode_steps)
                    total_episodes += 1
                    ep_rewards[i] = 0.0
                    ep_steps_above[i] = 0
                    ep_lengths[i] = 0
                    next_obs[i], _ = envs[i].reset()

            obs_arr = next_obs
            total_env_steps += args.n_envs

            # Gradient updates: one per env step after warmup.
            if total_env_steps >= args.warmup_steps:
                for _ in range(args.updates_per_step * args.n_envs):
                    diag = agent.update()
                    if diag:
                        last_diag = diag
                        total_updates += 1

            # Curriculum check (once per "period" env steps).
            if (len(reward_window) >= args.window
                    and total_env_steps % args.curriculum_check_every == 0):
                window_avg_reward = float(np.mean(reward_window))
                window_time_above = float(np.mean(time_above_window))
                previous_best = best_avg_reward
                if window_avg_reward > best_avg_reward:
                    best_avg_reward = window_avg_reward
                required_time_above = difficulty * 0.90
                if (window_time_above > required_time_above
                        and window_avg_reward > previous_best
                        and difficulty < 1.0):
                    delta_R = window_avg_reward - max(previous_best, 0.0)
                    step = max(args.ratchet_min, min(
                        args.ratchet_max,
                        args.ratchet_max * (delta_R / max(1.0, abs(previous_best) + 100.0)),
                    ))
                    difficulty = min(difficulty + step, 1.0)
                    _set_curriculum_all(difficulty)
                    print(f"[ratchet] env_steps={total_env_steps} d={difficulty:.3f} "
                          f"step={step:.3f} R={window_avg_reward:.1f} "
                          f"TimeUp={window_time_above*100:.1f}%")

            # Periodic console + CSV summary.
            if total_env_steps % args.log_interval == 0:
                avg_r = float(np.mean(reward_window)) if reward_window else 0.0
                avg_t = float(np.mean(time_above_window)) if time_above_window else 0.0
                threshold_deg = float(np.rad2deg(envs[0].reward_strategy.reward_threshold))
                csv_writer.writerow([
                    total_env_steps, total_updates, total_episodes, f"{avg_r:.4f}",
                    f"{float(np.mean(ep_lengths)):.2f}",
                    f"{difficulty:.4f}",
                    f"{envs[0].g:.3f}", f"{envs[0].friction_cart:.3f}",
                    f"{threshold_deg:.2f}",
                    f"{last_diag.get('critic_loss', 0):.4f}",
                    f"{last_diag.get('actor_loss', 0):.4f}",
                    f"{last_diag.get('alpha_loss', 0):.4f}",
                    f"{last_diag.get('alpha', 0):.4f}",
                    f"{last_diag.get('log_prob_mean', 0):.4f}",
                ])
                csv_file.flush()
                if tb_writer:
                    tb_writer.add_scalar("rollout/reward_mean", avg_r, total_env_steps)
                    tb_writer.add_scalar("rollout/time_above_mean", avg_t, total_env_steps)
                    tb_writer.add_scalar("curriculum/difficulty", difficulty, total_env_steps)
                    for k, v in last_diag.items():
                        tb_writer.add_scalar(f"sac/{k}", v, total_env_steps)
                if total_env_steps % args.console_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[step {total_env_steps:8d}] ep={total_episodes:5d} "
                        f"d={difficulty:.2f} R={avg_r:7.1f} "
                        f"TimeUp={avg_t*100:5.1f}% "
                        f"alpha={last_diag.get('alpha', 0):.3f} "
                        f"critic={last_diag.get('critic_loss', 0):.2f} "
                        f"actor={last_diag.get('actor_loss', 0):.2f} "
                        f"upd={total_updates} ({elapsed:.0f}s)"
                    )

            # Periodic checkpoint.
            if total_env_steps % args.save_interval == 0 and total_env_steps > 0:
                save_path = os.path.join(log_dir, f"sac_{run_name}_step{total_env_steps}.pth")
                agent.save(save_path, extra={"obs_rms": obs_rms.state_dict(),
                                             "difficulty": difficulty})

    except KeyboardInterrupt:
        print("[interrupt] saving checkpoint and exiting.")
    finally:
        final_path = os.path.join(log_dir, f"sac_{run_name}_final.pth")
        agent.save(final_path, extra={"obs_rms": obs_rms.state_dict(),
                                      "difficulty": difficulty})
        csv_file.close()
        if tb_writer:
            tb_writer.close()
        print(f"[done]   final checkpoint: {final_path}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--env", type=str, default="double", choices=["single", "double"])
    p.add_argument("--control", type=str, default="velocity", choices=["force", "velocity"])
    p.add_argument("--reward", type=str, default="hybrid",
                   choices=["standard", "exponential", "energy", "lqr", "hybrid"])
    p.add_argument("--reset_mode", type=str, default="down")
    p.add_argument("--integrator", type=str, default="rk4")
    p.add_argument("--x_soft", type=float, default=3.5)
    p.add_argument("--x_hard", type=float, default=10.0)
    p.add_argument("--boundary_penalty_k", type=float, default=0.1)
    p.add_argument("--wind_max", type=float, default=1.0)
    p.add_argument("--survival_alpha", type=float, default=0.0)
    p.add_argument("--threshold_curve", type=str, default="concave",
                   choices=["linear", "concave"])

    # SAC core.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--replay_capacity", type=int, default=1_000_000)
    p.add_argument("--warmup_steps", type=int, default=10_000)
    p.add_argument("--updates_per_step", type=int, default=1,
                   help="Gradient updates per env step PER ENV.")

    # Training scale.
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--total_env_steps", type=int, default=10_000_000)
    p.add_argument("--episode_steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=None)

    # Curriculum.
    p.add_argument("--start_difficulty", type=float, default=0.0)
    p.add_argument("--ratchet_min", type=float, default=0.005)
    p.add_argument("--ratchet_max", type=float, default=0.05)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--curriculum_check_every", type=int, default=400,
                   help="Evaluate ratchet gate every N env steps.")

    # Logging.
    p.add_argument("--log_interval", type=int, default=400)
    p.add_argument("--console_interval", type=int, default=4000)
    p.add_argument("--save_interval", type=int, default=100_000)

    p.add_argument("--load", type=str, default=None)
    return p


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    train(args)
