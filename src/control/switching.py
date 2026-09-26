r"""
Model-based switching between the four equilibria of the cart double pendulum
(Phase T; the model-based route to Phase 6).

Equilibria (pole angles from the downward vertical, cart at rest at
:math:`x = 0`) — indices match :class:`DoublePendulumGoalEnv`:

====  =========  ======================  =========
 idx   name       :math:`(\theta_1, \theta_2)`   stability
====  =========  ======================  =========
 0     DD         :math:`(0, 0)`          stable
 1     UU         :math:`(\pi, \pi)`      unstable
 2     DU         :math:`(0, \pi)`        unstable
 3     UD         :math:`(\pi, 0)`        unstable
====  =========  ======================  =========

Every ordered pair is connected by a trajectory from
:func:`src.control.swingup.optimize_swingup` (same multiple-shooting
transcription as the Phase R swing-up, with ``start`` and ``goal`` set to the
two equilibria). For a pole that must turn over, both directions
(:math:`\pm\pi`) are tried; the cheapest feasible solution is kept. Each
equilibrium is held by its own discrete infinite-horizon LQR, and each
transition is tracked with TVLQR whose terminal cost is the target's LQR cost.

:class:`SwitchingController` runs ``hold`` (LQR at the current equilibrium)
until :py:meth:`~SwitchingController.request` names a new target, then
``track`` along the stored trajectory, then ``hold`` at the target. The
library is built once by :func:`build_library` (``tools/eval_switching.py
--rebuild``) and stored in ``src/control/data/switching_d1.npz``.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np

from src.control.swingup import (
    PlantParams,
    SwingUpTrajectory,
    is_feasible,
    lqr_gain_at,
    optimize_swingup,
    state_error,
    tvlqr_gains,
)

NAMES = ("DD", "UU", "DU", "UD")
ANGLES = {"DD": (0.0, 0.0), "UU": (np.pi, np.pi), "DU": (0.0, np.pi), "UD": (np.pi, 0.0)}
DEFAULT_LIBRARY = os.path.join(os.path.dirname(__file__), "data", "switching_d1.npz")


def equilibrium(name: str) -> np.ndarray:
    t1, t2 = ANGLES[name]
    return np.array([0.0, t1, t2, 0.0, 0.0, 0.0])


def goal_candidates(src: str, dst: str) -> list[np.ndarray]:
    """Terminal states to try for ``src -> dst``: a pole that turns over may go
    either way (:math:`\\pm\\pi`); a pole that keeps its orientation stays put."""
    s = equilibrium(src)
    opts = []
    for i in (1, 2):
        d = ANGLES[dst][i - 1] - ANGLES[src][i - 1]
        opts.append([s[i]] if abs(d) < 1e-9 else [s[i] + np.pi, s[i] - np.pi])
    out = []
    for a1, a2 in itertools.product(*opts):
        g = np.zeros(6)
        g[1], g[2] = a1, a2
        out.append(g)
    return out


def plan_transition(p: PlantParams, src: str, dst: str, durations=(4.0, 5.0, 3.0),
                    verbose: bool = False, **kwargs) -> SwingUpTrajectory:
    """Cheapest feasible ``src -> dst`` trajectory at the first horizon that has one."""
    start = equilibrium(src)
    for T in durations:
        best = None
        for goal in goal_candidates(src, dst):
            traj = optimize_swingup(p, duration=T, start=start, goal=goal, **kwargs)
            ok = is_feasible(traj)
            if verbose:
                print(f"  {src}->{dst} T={T} goal=({goal[1]:+.2f},{goal[2]:+.2f}) "
                      f"feasible={ok} cost={traj.info['cost']:.4f}", flush=True)
            if ok and (best is None or traj.info["cost"] < best.info["cost"]):
                best = traj
        if best is not None:
            best.info.update({"src": src, "dst": dst})
            return best
    raise RuntimeError(f"No feasible trajectory for {src} -> {dst}.")


def build_library(p: PlantParams, pairs=None, verbose: bool = False
                  ) -> dict[tuple[str, str], SwingUpTrajectory]:
    pairs = pairs or [(a, b) for a in NAMES for b in NAMES if a != b]
    return {(a, b): plan_transition(p, a, b, verbose=verbose) for a, b in pairs}


def save_library(lib: dict[tuple[str, str], SwingUpTrajectory], path: str | Path) -> None:
    p = next(iter(lib.values())).params
    arrays = {"params": np.array([p.M, p.m1, p.m2, p.l1, p.l2, p.g,
                                  p.friction_cart, p.friction_pole, p.dt])}
    for (a, b), t in lib.items():
        arrays[f"{a}_{b}_states"] = t.states
        arrays[f"{a}_{b}_forces"] = t.forces
        arrays[f"{a}_{b}_goal"] = t.goal
    np.savez_compressed(path, **arrays)


def load_library(path: str | Path = DEFAULT_LIBRARY) -> dict[tuple[str, str], SwingUpTrajectory]:
    d = np.load(path)
    p = PlantParams(*map(float, d["params"]))
    lib = {}
    for a in NAMES:
        for b in NAMES:
            if f"{a}_{b}_states" in d:
                lib[(a, b)] = SwingUpTrajectory(states=d[f"{a}_{b}_states"],
                                                forces=d[f"{a}_{b}_forces"],
                                                goal=d[f"{a}_{b}_goal"], params=p)
    return lib


class SwitchingController:
    r"""
    Hold an equilibrium; on :py:meth:`request`, track the stored trajectory to
    the requested one and hold it there.

    A request is accepted only while holding and once the state has settled
    near the current equilibrium (every component within ``ready_tol``,
    angles wrapped), so each trajectory starts from its nominal initial state;
    until then it is queued. The controller outputs newtons; use
    :py:meth:`action` for :class:`ForceControl`.
    """

    def __init__(self, library: dict[tuple[str, str], SwingUpTrajectory],
                 Q: np.ndarray | None = None, R: float = 0.01, force_limit: float = 500.0,
                 ready_tol: float = 0.05, initial: str = "DD") -> None:
        self.lib = library
        self.params = next(iter(library.values())).params
        self.K_hold = {n: lqr_gain_at(self.params, equilibrium(n), Q, R)[0] for n in NAMES}
        P_hold = {n: lqr_gain_at(self.params, equilibrium(n), Q, R)[1] for n in NAMES}
        self.K_track = {k: tvlqr_gains(t, Q, R, P_final=P_hold[k[1]]) for k, t in library.items()}
        self.force_limit = float(force_limit)
        self.ready_tol = float(ready_tol)
        self.reset(initial)

    def reset(self, initial: str = "DD") -> None:
        self.current = initial
        self.target = initial
        self.phase = "hold"
        self._traj: SwingUpTrajectory | None = None
        self._K = None
        self._k = 0
        self._dst = initial

    def request(self, target: str) -> None:
        if target not in NAMES:
            raise ValueError(f"Unknown equilibrium {target!r}; expected one of {NAMES}.")
        self.target = target

    @property
    def settled(self) -> bool:
        return self.phase == "hold" and self.current == self.target

    def force(self, state: np.ndarray) -> float:
        s = np.asarray(state, dtype=np.float64)
        if self.phase == "hold" and self.target != self.current:
            err = state_error(s, equilibrium(self.current))
            if np.max(np.abs(err)) < self.ready_tol:
                key = (self.current, self.target)
                self._traj, self._K, self._k = self.lib[key], self.K_track[key], 0
                self._dst = self.target
                self.phase = "track"
        if self.phase == "track":
            if self._k < len(self._traj.forces):
                j = self._k
                self._k += 1
                err = state_error(s, self._traj.states[j])
                return self._clip(float(self._traj.forces[j] - (self._K[j] @ err)[0]))
            self.current = self._dst
            self.phase = "hold"
        err = state_error(s, equilibrium(self.current))
        return self._clip(float(-(self.K_hold[self.current] @ err)[0]))

    def action(self, state: np.ndarray, max_force: float = 5000.0) -> np.ndarray:
        return np.array([self.force(state) / max_force], dtype=np.float32)

    def _clip(self, f: float) -> float:
        return float(np.clip(f, -self.force_limit, self.force_limit))
