r"""
Closed-loop evaluation of the designed swing-up controller
(:mod:`src.control.swingup`) on the actual environment.

Runs ``--episodes`` episodes of ``--horizon`` seconds from the environment's
``down`` reset (hanging, :math:`\pm 0.05` uniform noise on every state) at
curriculum difficulty ``--difficulty`` (:math:`\delta = 1`: full gravity,
frictionless, wind :math:`\sigma_w = 1` N), driving the cart through
:class:`ForceControl`. Reports the honest balance metrics introduced in
Phase O:

* **survived** — the episode never hit the hard cart bound;
* **in soft bound** — the cart never left :math:`|x| \le x_{\rm soft}`;
* **terminal-1 s strict** — fraction of the final second with both poles
  within 0.17 rad (≈ 10°) of upright;
* **max sustained strict** — longest unbroken strict dwell;
* **steady-state P1/P2** — mean absolute pole error over the final second.

Usage::

    python tools/eval_swingup.py --difficulty 1.0 --episodes 20
    python tools/eval_swingup.py --report docs/reports/swingup_d1.md

The nominal trajectory is loaded from ``--traj`` (default: the committed
``src/control/data/swingup_d1.npz``) when its physics match the requested
difficulty, and re-optimised otherwise (``--reoptimize`` forces this).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.control.swingup import (  # noqa: E402
    PlantParams,
    SwingUpController,
    SwingUpTrajectory,
    best_swingup,
    wrap_angle,
)
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402

DEFAULT_TRAJ = os.path.join(os.path.dirname(__file__), "..", "src", "control", "data",
                            "swingup_d1.npz")
STRICT = 0.17


def load_or_optimize(p: PlantParams, path: str, reoptimize: bool, verbose: bool = True
                     ) -> SwingUpTrajectory:
    if not reoptimize and os.path.exists(path):
        traj = SwingUpTrajectory.load(path)
        if np.allclose(np.array(list(vars(traj.params).values())),
                       np.array(list(vars(p).values()))):
            return traj
        if verbose:
            print(f"{path} was optimised for different physics; re-optimising.")
    t0 = time.time()
    traj = best_swingup(p, verbose=verbose)
    if verbose:
        print(f"Trajectory optimisation: {time.time() - t0:.0f} s, "
              f"T = {traj.duration:.2f} s, goal = {traj.goal[1:3]}")
    return traj


def run_episode(env: DoublePendulumCartEnv, ctrl: SwingUpController, seed: int,
                horizon: float, reset_mode: str = "down") -> dict:
    env.reset(seed=seed, options={"mode": reset_mode})
    ctrl.reset(env.state)
    n = int(round(horizon / env.dt))
    errs = np.zeros((n, 2))
    max_force = 0.0
    left_soft = False
    survived = True
    for j in range(n):
        a = ctrl.action(env.state, env.control_strategy.max_force)
        max_force = max(max_force, abs(float(a[0])) * env.control_strategy.max_force)
        _, _, terminated, _, _ = env.step(a)
        errs[j] = np.abs(wrap_angle(env.state[1:3] - np.pi))
        left_soft |= abs(float(env.state[0])) > env.x_soft
        if terminated:
            survived = False
            errs[j + 1:] = np.pi
            break
    strict = np.all(errs < STRICT, axis=1)
    last = int(round(1.0 / env.dt))
    runs, cur = 0, 0
    for ok in strict:
        cur = cur + 1 if ok else 0
        runs = max(runs, cur)
    first = int(np.argmax(strict)) if strict.any() else -1
    return {
        "survived": survived,
        "in_soft": survived and not left_soft,
        "terminal_strict": float(strict[-last:].mean()),
        "max_sustained": runs * env.dt,
        "first_strict": first * env.dt if first >= 0 else float("nan"),
        "ss_err": np.degrees(errs[-last:].mean(axis=0)),
        "max_force": max_force,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset_mode", default="down", choices=["down", "up"])
    ap.add_argument("--wind", type=float, default=None,
                    help="pin wind std [N] (default: curriculum value)")
    ap.add_argument("--traj", default=DEFAULT_TRAJ)
    ap.add_argument("--reoptimize", action="store_true")
    ap.add_argument("--save_traj", default=None, help="write the trajectory used here")
    ap.add_argument("--report", default=None, help="write a markdown report here")
    args = ap.parse_args()

    env = DoublePendulumCartEnv(control_strategy=ForceControl())
    if args.wind is not None:
        env.set_wind_pinned(args.wind)
    env.set_curriculum(args.difficulty)
    p = PlantParams.from_env(env)
    traj = load_or_optimize(p, args.traj, args.reoptimize)
    if args.save_traj:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_traj)), exist_ok=True)
        traj.save(args.save_traj)
    ctrl = SwingUpController(traj)

    rows = []
    t0 = time.time()
    for ep in range(args.episodes):
        r = run_episode(env, ctrl, args.seed + ep, args.horizon, args.reset_mode)
        rows.append(r)
        print(f"ep {ep:3d}  survived={r['survived']!s:5}  term1s={100 * r['terminal_strict']:6.1f}%  "
              f"sustained={r['max_sustained']:6.2f}s  first={r['first_strict']:5.2f}s  "
              f"ss=({r['ss_err'][0]:.2f}°, {r['ss_err'][1]:.2f}°)  |F|max={r['max_force']:.0f} N")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    ss = np.mean([r["ss_err"] for r in rows], axis=0)
    summary = [
        ("Episodes", f"{len(rows)}"),
        ("Survived", f"{100 * mean('survived'):.1f} %"),
        ("Stayed within soft cart bound", f"{100 * mean('in_soft'):.1f} %"),
        ("Terminal-1 s strict (mean)", f"{100 * mean('terminal_strict'):.1f} %"),
        ("Episodes with terminal-1 s strict = 100 %",
         f"{sum(r['terminal_strict'] == 1.0 for r in rows)} / {len(rows)}"),
        ("Max sustained strict (mean)", f"{mean('max_sustained'):.2f} s of {args.horizon:.0f} s"),
        ("First strict entry (mean)", f"{np.nanmean([r['first_strict'] for r in rows]):.2f} s"),
        ("Steady-state P1 / P2", f"{ss[0]:.3f}° / {ss[1]:.3f}°"),
        ("Peak cart force (max)", f"{max(r['max_force'] for r in rows):.0f} N"),
    ]
    print(f"\n{len(rows)} episodes in {time.time() - t0:.0f} s")
    for k, v in summary:
        print(f"  {k:45s} {v}")

    if args.report:
        info = traj.info
        lines = [
            f"# Designed swing-up evaluation (δ = {args.difficulty:g})",
            "",
            f"Reset `{args.reset_mode}`, horizon {args.horizon:g} s, wind σ = {env.wind_std:g} N, "
            f"seeds {args.seed}–{args.seed + args.episodes - 1}. Strict band: both poles within "
            f"{STRICT} rad of upright.",
            "",
            f"Trajectory: T = {traj.duration:.2f} s, goal angles = "
            f"({traj.goal[1]:+.4f}, {traj.goal[2]:+.4f}), peak nominal force "
            f"{np.abs(traj.forces).max():.1f} N, peak nominal |x| "
            f"{np.abs(traj.states[:, 0]).max():.2f} m"
            + (f", max defect {info['max_defect']:.1e}" if "max_defect" in info else "") + ".",
            "",
            "| Metric | Value |",
            "|---|---:|",
            *[f"| {k} | {v} |" for k, v in summary],
            "",
            "Command:",
            "",
            "```bash",
            "python tools/eval_swingup.py " + " ".join(sys.argv[1:]),
            "```",
            "",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            f.write("\n".join(lines))
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
