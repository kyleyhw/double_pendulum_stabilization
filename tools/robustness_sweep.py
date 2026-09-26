r"""
Phase 4.7 stress test of the Phase R controller (``src/control/swingup.py``).

Every run uses the real ``DoublePendulumCartEnv`` at full difficulty
(:math:`\delta = 1`, wind :math:`\sigma_w = 1` N unless swept) driven through
``ForceControl``. A run *succeeds* if the cart never leaves
:math:`|x| \le x_{\rm soft} = 3.5` m and both poles are within 0.17 rad
(≈ 10°) of upright for the whole final second.

Sweeps
------
A. **Angle basin** — balance controller from rest at
   :math:`(\pi + \Delta\theta_1, \pi + \Delta\theta_2)`, 10 s.
B. **Rate basin** — balance controller from upright with
   :math:`(\dot\theta_1, \dot\theta_2)`, 10 s.
C. **Impulse** — full swing-up; a one-step (5 ms) push of impulse
   :math:`J` on the cart either while balancing (2 s after the swing ends)
   or mid-swing (2 s into the trajectory).
D. **Wind** — full swing-up under wind :math:`\sigma_w`.
E. **Model mismatch** — full swing-up with the plant's mass / length /
   friction perturbed; the controller keeps the nominal model.
F. **Actuator limit** — full swing-up with the controller's force clipped.
G. **Gain trade-off** — the LQR control weight :math:`R` (default 0.01)
   against basin size and wind rejection.

Usage::

    python tools/robustness_sweep.py                 # writes docs/robustness_report.md
    python tools/robustness_sweep.py --quick         # coarse grids, for smoke tests
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.control.swingup import SwingUpController, SwingUpTrajectory, wrap_angle  # noqa: E402
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402
from tools.eval_swingup import DEFAULT_TRAJ  # noqa: E402

STRICT = 0.17
REPORT = os.path.join(os.path.dirname(__file__), "..", "docs", "robustness_report.md")
FIGURE = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "robustness_basins.png")


def make_env(wind: float | None = None, **overrides) -> DoublePendulumCartEnv:
    env = DoublePendulumCartEnv(control_strategy=ForceControl())
    if wind is not None:
        env.set_wind_pinned(wind)
    env.set_curriculum(1.0)
    for k, v in overrides.items():
        setattr(env, k, v)
    return env


def rollout(env, ctrl, seconds: float, impulse_step: int | None = None,
            impulse: float = 0.0) -> bool:
    """Step until ``seconds`` elapse; True on success (see module docstring)."""
    n = int(round(seconds / env.dt))
    last = int(round(1.0 / env.dt))
    max_force = env.control_strategy.max_force
    for j in range(n):
        if impulse_step is not None and j == impulse_step:
            env.apply_impulse(impulse / env.dt)
        env.step(ctrl.action(env.state, max_force))
        if abs(env.state[0]) > env.x_soft:
            return False
        if j >= n - last and np.any(np.abs(wrap_angle(env.state[1:3] - np.pi)) >= STRICT):
            return False
    return True


def balance_from(env, ctrl, state: np.ndarray, seed: int, seconds: float = 10.0) -> bool:
    env.reset(seed=seed)
    env.state = np.asarray(state, dtype=np.float64).copy()
    ctrl.reset()
    ctrl.phase = "balance"
    return rollout(env, ctrl, seconds)


def swing_up(env, ctrl, seed: int, horizon: float = 20.0, impulse_at: float | None = None,
             impulse: float = 0.0, relative_to: str = "balance") -> bool:
    """Full episode from the ``down`` reset. ``impulse_at`` is measured from the
    start of the swing (``relative_to='swing'``) or of balance (``'balance'``);
    the settle phase is shortened to its minimum so these times are fixed."""
    env.reset(seed=seed, options={"mode": "down"})
    ctrl.reset(env.state)
    step = None
    if impulse_at is not None:
        settle = ctrl.settle_steps
        offset = settle if relative_to == "swing" else settle + len(ctrl.traj.forces)
        step = offset + int(round(impulse_at / env.dt))
    return rollout(env, ctrl, horizon, impulse_step=step, impulse=impulse)


def success_rate(fn, seeds) -> float:
    return float(np.mean([fn(s) for s in seeds]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--figure", default=FIGURE)
    args = ap.parse_args()
    t0 = time.time()
    traj = SwingUpTrajectory.load(DEFAULT_TRAJ)
    seeds = range(args.seeds if not args.quick else 3)
    # Fixed-duration settle so impulse timings are comparable across seeds.
    ctrl_kw = dict(settle_time=1.0, min_settle_time=1.0)

    def controller(**kw) -> SwingUpController:
        return SwingUpController(traj, **{**ctrl_kw, **kw})

    env = make_env()
    ctrl = controller()

    # --- A/B: basins -------------------------------------------------------
    step_a, lim_a = (0.1, 0.8) if args.quick else (0.04, 0.8)
    ang = np.round(np.arange(-lim_a, lim_a + 1e-9, step_a), 4)
    basin_ang = np.zeros((len(ang), len(ang)))
    for i, d1 in enumerate(ang):
        for j, d2 in enumerate(ang):
            s = np.array([0.0, np.pi + d1, np.pi + d2, 0.0, 0.0, 0.0])
            basin_ang[i, j] = balance_from(env, ctrl, s, seed=i * 1000 + j)
    print(f"A angle basin done ({time.time() - t0:.0f} s)", flush=True)

    step_b, lim_b = (0.5, 4.0) if args.quick else (0.2, 4.0)
    rate = np.round(np.arange(-lim_b, lim_b + 1e-9, step_b), 4)
    basin_rate = np.zeros((len(rate), len(rate)))
    for i, w1 in enumerate(rate):
        for j, w2 in enumerate(rate):
            s = np.array([0.0, np.pi, np.pi, 0.0, w1, w2])
            basin_rate[i, j] = balance_from(env, ctrl, s, seed=i * 1000 + j)
    print(f"B rate basin done ({time.time() - t0:.0f} s)", flush=True)

    def radius(grid, axis_vals, same_sign: bool) -> float:
        """Largest r such that every grid point with max(|a|,|b|) <= r succeeds."""
        best = 0.0
        for r in sorted({abs(v) for v in axis_vals}):
            mask = np.maximum.outer(np.abs(axis_vals), np.abs(axis_vals)) <= r + 1e-9
            if grid[mask].all():
                best = r
            else:
                break
        return best

    r_ang = radius(basin_ang, ang, True)
    r_rate = radius(basin_rate, rate, True)
    diag = [basin_ang[i, i] for i in range(len(ang))]
    anti = [basin_ang[i, len(ang) - 1 - i] for i in range(len(ang))]

    def edge(vals, flags):
        ok = [abs(v) for v, f in zip(vals, flags, strict=True) if f]
        bad = [abs(v) for v, f in zip(vals, flags, strict=True) if not f]
        lim = min(bad) if bad else float("inf")
        return max([v for v in ok if v < lim], default=0.0)

    # --- C: impulse --------------------------------------------------------
    impulses = [0.5, 1, 2, 3, 4, 5, 6, 8, 10] if not args.quick else [1, 4, 10]
    imp_bal = [success_rate(lambda s, J=J: swing_up(env, ctrl, s, impulse_at=2.0, impulse=J), seeds)
               for J in impulses]
    imp_swing = [success_rate(lambda s, J=J: swing_up(env, ctrl, s, impulse_at=2.0, impulse=J,
                                                      relative_to="swing"), seeds)
                 for J in impulses]
    print(f"C impulse done ({time.time() - t0:.0f} s)", flush=True)

    # --- D: wind -----------------------------------------------------------
    winds = [1, 5, 10, 20, 30, 40, 60] if not args.quick else [1, 20]
    wind_rows = []
    for w in winds:
        e = make_env(wind=w)
        wind_rows.append(success_rate(lambda s, e=e: swing_up(e, ctrl, s), seeds))
    print(f"D wind done ({time.time() - t0:.0f} s)", flush=True)

    # --- E: model mismatch -------------------------------------------------
    nominal = make_env()
    mism = []
    factors = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3] if not args.quick else [0.8, 1.2]
    for name in ("M", "m1", "m2", "l1", "l2"):
        row = []
        for f in factors:
            e = make_env(**{name: getattr(nominal, name) * f})
            row.append(success_rate(lambda s, e=e: swing_up(e, ctrl, s), seeds))
        mism.append((name, row))
    frictions = [0.05, 0.1, 0.2, 0.5] if not args.quick else [0.1]
    fric_rows = []
    for mu in frictions:
        e = make_env(friction_cart=mu)
        fric_rows.append(("cart", mu, success_rate(lambda s, e=e: swing_up(e, ctrl, s), seeds)))
    for mu in [0.01, 0.02, 0.05] if not args.quick else [0.02]:
        e = make_env(friction_pole=mu)
        fric_rows.append(("pole", mu, success_rate(lambda s, e=e: swing_up(e, ctrl, s), seeds)))
    print(f"E mismatch done ({time.time() - t0:.0f} s)", flush=True)

    # --- F: actuator limit -------------------------------------------------
    limits = [20, 25, 27, 30, 35, 40, 60] if not args.quick else [25, 40]
    lim_ctrls = [controller(force_limit=L) for L in limits]
    lim_rows = [success_rate(lambda s, c=c: swing_up(env, c, s), seeds) for c in lim_ctrls]
    print(f"F actuator done ({time.time() - t0:.0f} s)", flush=True)

    # --- G: gain trade-off -------------------------------------------------
    g_ang = np.linspace(-0.5, 0.5, 11 if not args.quick else 5)
    gains = [0.01, 1.0, 3.0]
    gain_rows = []
    for R in gains:
        c = controller(R=R)
        cells = [(a, b) for a in g_ang for b in g_ang]
        basin = np.mean([balance_from(env, c, np.array([0, np.pi + a, np.pi + b, 0, 0, 0.0]), k)
                         for k, (a, b) in enumerate(cells)])
        w_env = make_env(wind=40.0)
        wind40 = success_rate(lambda s, c=c, e=w_env: swing_up(e, c, s), seeds)
        gain_rows.append((R, basin, wind40, np.abs(c.K_balance).max()))
    print(f"G gains done ({time.time() - t0:.0f} s)", flush=True)

    plot_basins(ang, basin_ang, rate, basin_rate, args.figure)
    write_report(args, locals(), time.time() - t0)


def plot_basins(ang, basin_ang, rate, basin_rate, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fail, ok = "#e8e7e3", "#256abf"  # neutral surface-adjacent gray / blue-500
    cmap = ListedColormap([fail, ok])
    ink, muted = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor="#fcfcfb")
    panels = [
        (axes[0], np.degrees(ang), basin_ang, "Initial angle error, pole 1 (deg)",
         "Initial angle error, pole 2 (deg)", "A. Angle basin (from rest)"),
        (axes[1], rate, basin_rate, "Initial rate, pole 1 (rad/s)",
         "Initial rate, pole 2 (rad/s)", "B. Rate basin (from upright)"),
    ]
    for ax, v, grid, xl, yl, title in panels:
        step = v[1] - v[0]
        ext = [v[0] - step / 2, v[-1] + step / 2, v[0] - step / 2, v[-1] + step / 2]
        # grid[i, j]: i indexes pole 1 -> x axis, so transpose for imshow rows = y.
        ax.imshow(grid.T, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        ax.set_facecolor("#fcfcfb")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.legend(handles=[Patch(color=ok, label="Recovered (strict for final 1 s)"),
                        Patch(color=fail, label="Failed")],
               loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Phase R balance controller: recoverable initial states at δ = 1 "
                 "(wind σ = 1 N, 10 s)", x=0.02, ha="left", color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def write_report(args, v: dict, wall: float) -> None:
    n = len(v["seeds"])
    ang, rate = v["ang"], v["rate"]
    lines = [
        "# Robustness report — Phase R swing-up + balance controller",
        "",
        "Stress test of `src/control/swingup.py` (Phase 4.7 of `PROJECT_PLAN.md`), "
        "generated by `python tools/robustness_sweep.py`"
        + (" --quick" if args.quick else "") + f" in {wall / 60:.0f} min.",
        "",
        "All runs use the real `DoublePendulumCartEnv` at full difficulty (δ = 1: "
        "g = 9.81 m/s², frictionless, wind σ = 1 N per 5 ms step unless swept) driven "
        "through `ForceControl`. A run **succeeds** if the cart stays within "
        "|x| ≤ 3.5 m throughout and both poles are within 0.17 rad (≈ 10°) of upright "
        "for the whole final second. Swing-up runs last 20 s from the `down` reset "
        f"(1 s settle, 4 s swing, 15 s balance); rates are over {n} seeds.",
        "",
        "## Headline numbers",
        "",
        "| Quantity | Recoverable |",
        "|---|---:|",
        f"| Angle error, both poles, any sign combination (from rest) | ±{np.degrees(v['r_ang']):.1f}° |",
        f"| Angle error, same sign (Δθ₁ = Δθ₂) | ±{np.degrees(v['edge'](ang, v['diag'])):.1f}° |",
        f"| Angle error, opposite sign (Δθ₁ = −Δθ₂) | ±{np.degrees(v['edge'](ang, v['anti'])):.1f}° |",
        f"| Angular rate, both poles, any combination (from upright) | ±{v['r_rate']:.1f} rad/s |",
        f"| Impulse on the cart while balancing (100 % of seeds) | "
        f"{max([J for J, r in zip(v['impulses'], v['imp_bal'], strict=True) if r == 1.0], default=0):g} N·s |",
        f"| Impulse on the cart mid-swing (100 % of seeds) | "
        f"{max([J for J, r in zip(v['impulses'], v['imp_swing'], strict=True) if r == 1.0], default=0):g} N·s |",
        f"| Wind σ (100 % of seeds) | "
        f"{max([w for w, r in zip(v['winds'], v['wind_rows'], strict=True) if r == 1.0], default=0):g} N |",
        "",
        "The grid resolution is "
        f"{np.degrees(ang[1] - ang[0]):.1f}° for angles and {rate[1] - rate[0]:.1f} rad/s for "
        "rates, so basin radii are lower bounds to within one grid step. Impulse "
        "and wind limits are the largest *tested* values with 100 % success.",
        "",
        "![Recoverable initial states](images/robustness_basins.png)",
        "",
        "How to read the basins: each cell is one 10 s balance run starting at rest "
        "(A) or upright (B). Both basins are sheared parallelograms, not discs. In A, "
        "errors in which both poles lean the same way (the chain stays straight) are "
        "tolerated about twice as far as errors that bend the chain, which the cart "
        "can only correct by moving one way and then the other. In B the basin is a "
        "narrow band: large pole-1 rates are recoverable when pole 2 turns the other "
        "way at roughly a quarter of the rate, but a few tenths of a rad/s in the "
        "worst direction already fail. The swing-up hands over well inside both "
        "basins (errors < 0.1° at the end of the nominal), which is why the "
        "controller is reliable from hanging despite a small balance basin.",
        "",
        "## C. Impulsive push on the cart",
        "",
        "One 5 ms force pulse of impulse J (force J/0.005 s), 2 s after the swing "
        "ends (balancing) or 2 s into the 4 s swing.",
        "",
        "| J [N·s] | " + " | ".join(f"{J:g}" for J in v["impulses"]) + " |",
        "|---|" + "---:|" * len(v["impulses"]),
        "| While balancing | " + " | ".join(pct(r) for r in v["imp_bal"]) + " |",
        "| Mid-swing | " + " | ".join(pct(r) for r in v["imp_swing"]) + " |",
        "",
        "## D. Wind (full swing-up)",
        "",
        "| σ_w [N] | " + " | ".join(f"{w:g}" for w in v["winds"]) + " |",
        "|---|" + "---:|" * len(v["winds"]),
        "| Success | " + " | ".join(pct(r) for r in v["wind_rows"]) + " |",
        "",
        "## E. Model mismatch (full swing-up, controller keeps the nominal model)",
        "",
        "Plant parameter multiplied by the factor shown; nominal M = 1 kg, "
        "m₁ = m₂ = 0.5 kg, l₁ = l₂ = 1 m.",
        "",
        "| Parameter | " + " | ".join(f"×{f:g}" for f in v["factors"]) + " |",
        "|---|" + "---:|" * len(v["factors"]),
        *[f"| {name} | " + " | ".join(pct(r) for r in row) + " |" for name, row in v["mism"]],
        "",
        "| Unmodelled friction | Coefficient | Success |",
        "|---|---:|---:|",
        *[f"| {kind} | {mu:g} | {pct(r)} |" for kind, mu, r in v["fric_rows"]],
        "",
        "## F. Actuator limit (full swing-up)",
        "",
        "The nominal trajectory peaks at 26.4 N; feedback needs headroom above that.",
        "",
        "| Force limit [N] | " + " | ".join(f"{L:g}" for L in v["limits"]) + " |",
        "|---|" + "---:|" * len(v["limits"]),
        "| Success | " + " | ".join(pct(r) for r in v["lim_rows"]) + " |",
        "",
        "Reading E: cart mass and friction barely matter, because the cart is the "
        "actuated coordinate and feedback absorbs them. The pole parameters do "
        "matter: the swing is dominated by the feed-forward force profile, which is "
        "tuned to the chain's natural dynamics, and a ±10–20 % error in m₁, m₂ or l₁ "
        "(or even −10 % in l₂) puts the chain outside the TVLQR tube. Remedies, not "
        "implemented here: identify the parameters before planning, re-plan online "
        "(MPC), or optimise the trajectory over a set of parameter samples.",
        "",
        "## G. Gain trade-off (LQR control weight R)",
        "",
        "Same Q = diag(10, 100, 100, 1, 1, 1) throughout; R also sets the TVLQR "
        "tracking gains. Basin = fraction of an 11 × 11 grid of initial angle errors in "
        "±0.5 rad (±29°) recovered from rest; wind = full swing-up at σ = 40 N.",
        "",
        "| R | Largest gain | Basin fraction | Swing-up at σ = 40 N |",
        "|---:|---:|---:|---:|",
        *[f"| {R:g}{' (default)' if R == 0.01 else ''} | {k:.0f} | {pct(b)} | {pct(w)} |"
          for R, b, w, k in v["gain_rows"]],
        "",
        "High gain rejects disturbances; low gain tolerates larger initial errors "
        "(it is less violent, so the nonlinear chain stays closer to the linear "
        "model). The default favours disturbance rejection, since the swing-up "
        "delivers the chain to upright with errors well inside every basin here.",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
