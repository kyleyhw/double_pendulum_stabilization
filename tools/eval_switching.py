r"""
Build and evaluate the equilibrium-switching library (``src/control/switching.py``).

Per-transition test: the env starts at the source equilibrium plus uniform
:math:`\pm 0.02` noise on every state, the controller holds it for 1 s, the
target is requested, and the episode runs until 5 s after the nominal
transition ends. Success: the cart stays within :math:`|x| \le x_{\rm soft}`
and both poles are within 0.17 rad (≈ 10°) of the target configuration for the
whole final second.

Tour test: one continuous episode that visits every ordered pair once —
DD→UU→DD→DU→DD→UD→UU→DU→UU→UD→DU→UD→DD — requesting the next target 3 s
after the previous one is reached.

All runs at :math:`\delta = 1` (wind :math:`\sigma_w = 1` N) with ``ForceControl``.

Usage::

    python tools/eval_switching.py --rebuild         # re-optimise all 12 transitions
    python tools/eval_switching.py --seeds 10 --report docs/reports/switching_d1.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.control.swingup import PlantParams, wrap_angle  # noqa: E402
from src.control.switching import (  # noqa: E402
    DEFAULT_LIBRARY,
    NAMES,
    SwitchingController,
    equilibrium,
    load_library,
    plan_transition,
    save_library,
)
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402

STRICT = 0.17
TOUR = ["DD", "UU", "DD", "DU", "DD", "UD", "UU", "DU", "UU", "UD", "DU", "UD", "DD"]


def make_env() -> DoublePendulumCartEnv:
    env = DoublePendulumCartEnv(control_strategy=ForceControl())
    env.set_curriculum(1.0)
    return env


def _plan(pair):
    a, b = pair
    p = PlantParams.from_env(make_env())
    t0 = time.time()
    traj = plan_transition(p, a, b)
    print(f"{a}->{b}: T={traj.duration:.1f} s goal=({traj.goal[1]:+.2f},{traj.goal[2]:+.2f}) "
          f"peak F={np.abs(traj.forces).max():.1f} N ({time.time() - t0:.0f} s)", flush=True)
    return pair, traj


def at(state: np.ndarray, name: str) -> bool:
    return bool(np.all(np.abs(wrap_angle(state[1:3] - equilibrium(name)[1:3])) < STRICT))


def run_transition(env, ctrl, src: str, dst: str, seed: int) -> dict:
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    env.state = equilibrium(src) + rng.uniform(-0.02, 0.02, 6)
    ctrl.reset(src)
    T = ctrl.lib[(src, dst)].duration
    n = int(round((1.0 + T + 5.0) / env.dt))
    last = int(round(1.0 / env.dt))
    ok, in_soft, max_f, reached = True, True, 0.0, None
    for j in range(n):
        if j == int(round(1.0 / env.dt)):
            ctrl.request(dst)
        a = ctrl.action(env.state, env.control_strategy.max_force)
        max_f = max(max_f, abs(float(a[0])) * env.control_strategy.max_force)
        env.step(a)
        in_soft &= abs(env.state[0]) <= env.x_soft
        if reached is None and ctrl.settled and at(env.state, dst) and j > int(1.0 / env.dt):
            reached = (j + 1) * env.dt - 1.0
        if j >= n - last:
            ok &= at(env.state, dst)
    err = np.degrees(np.abs(wrap_angle(env.state[1:3] - equilibrium(dst)[1:3])))
    return {"success": bool(ok and in_soft), "reached": reached, "max_force": max_f,
            "final_err": err}


def run_tour(env, ctrl, seed: int, dwell: float = 3.0, limit: float = 120.0) -> dict:
    env.reset(seed=seed)
    env.state = equilibrium(TOUR[0]) + np.random.default_rng(seed).uniform(-0.02, 0.02, 6)
    ctrl.reset(TOUR[0])
    k, hold_steps, visited = 1, 0, []
    in_soft = True
    for _ in range(int(limit / env.dt)):
        env.step(ctrl.action(env.state, env.control_strategy.max_force))
        in_soft &= abs(env.state[0]) <= env.x_soft
        if ctrl.settled and at(env.state, ctrl.current):
            hold_steps += 1
        if hold_steps * env.dt >= dwell:
            visited.append(ctrl.current)
            if k == len(TOUR):
                break
            ctrl.request(TOUR[k])
            k += 1
            hold_steps = 0
    return {"completed": visited == TOUR, "legs": max(0, len(visited) - 1),
            "in_soft": in_soft}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--report", default=None)
    ap.add_argument("--policy", default=None,
                    help="evaluate a goal-conditioned RL policy (src/train_balance.py "
                         "--goals all) instead of the model-based controller")
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(args.library):
        pairs = [(a, b) for a in NAMES for b in NAMES if a != b]
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            lib = dict(ex.map(_plan, pairs))
        save_library(lib, args.library)
        print(f"Library ({len(lib)} transitions) built in {time.time() - t0:.0f} s -> {args.library}")
    lib = load_library(args.library)
    if args.policy:
        from tools.eval_balance import PolicySwitcher
        ctrl = PolicySwitcher(args.policy, lib)
        env = DoublePendulumCartEnv(control_strategy=ForceControl(max_force=ctrl.max_force))
        env.set_curriculum(1.0)
    else:
        ctrl = SwitchingController(lib)
        env = make_env()

    rows = []
    for (a, b), traj in lib.items():
        rs = [run_transition(env, ctrl, a, b, s) for s in range(args.seeds)]
        rate = np.mean([r["success"] for r in rs])
        reach = [r["reached"] for r in rs if r["reached"] is not None]
        rows.append((a, b, traj, rate, np.mean(reach) if reach else float("nan"),
                     max(r["max_force"] for r in rs),
                     np.mean([r["final_err"] for r in rs], axis=0)))
        print(f"{a}->{b}: success {100 * rate:.0f} %  reached {rows[-1][4]:.2f} s  "
              f"peak |F| {rows[-1][5]:.0f} N", flush=True)
    tours = [run_tour(env, ctrl, s) for s in range(max(1, args.seeds // 2))]
    n_tour = sum(t["completed"] and t["in_soft"] for t in tours)
    print(f"Tour: {n_tour}/{len(tours)} completed all {len(TOUR) - 1} legs")

    if args.report:
        lines = [
            "# Equilibrium switching evaluation (δ = 1)",
            "",
            (f"Goal-conditioned RL policy `{args.policy}` (`src/train_balance.py --goals all`); "
             "the trajectory columns describe the library it was trained along, not "
             "what the policy does. "
             if args.policy else
             "Model-based switching between the four equilibria (`src/control/switching.py`): "
             "one multiple-shooting trajectory per ordered pair, TVLQR tracking, LQR hold at "
             "each equilibrium. ")
            + "Env: `DoublePendulumCartEnv`, full difficulty, wind σ = 1 N, "
            f"`ForceControl`; {args.seeds} seeds per transition. Success = cart within "
            "|x| ≤ 3.5 m throughout and both poles within 0.17 rad of the target for the "
            "whole final second (5 s after the nominal transition ends).",
            "",
            "| From → to | Direction (Δθ₁, Δθ₂) | T [s] | Nominal peak F [N] | Success | "
            "Reached after [s] | Peak F [N] | Final error P1 / P2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for a, b, traj, rate, reach, mf, fe in rows:
            d = traj.goal[1:3] - traj.states[0, 1:3]
            dirs = ", ".join("0" if abs(x) < 1e-6 else ("+π" if x > 0 else "−π") for x in d)
            lines.append(f"| {a} → {b} | ({dirs}) | {traj.duration:.1f} | "
                         f"{np.abs(traj.forces).max():.1f} | {100 * rate:.0f} % | {reach:.2f} | "
                         f"{mf:.0f} | {fe[0]:.2f}° / {fe[1]:.2f}° |")
        lines += [
            "",
            f"**Tour** (all 12 transitions in one episode, 3 s dwell at each stop): "
            f"{n_tour} / {len(tours)} completed.",
            "",
            "\"Reached after\" is measured from the request to the first step at which the "
            "controller is holding the target with both poles in the strict band. "
            "Direction ±π is the way each pole turns over; 0 = keeps its orientation.",
            "",
            "Command: `python tools/eval_switching.py " + " ".join(sys.argv[1:]) + "`",
            "",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            f.write("\n".join(lines))
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
