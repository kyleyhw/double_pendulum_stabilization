r"""
Designed swing-up for the cart double pendulum: trajectory optimisation,
time-varying LQR tracking, and LQR balance.

Why this exists
===============
Energy-shaping swing-up (and every RL reward built on it) regulates the total
energy onto the up-up level :math:`E_0` and then waits for the chain to visit
the upright configuration. For one pole that level set *is* the homoclinic
orbit, so the wait is finite. For two poles the level set is 3-D and chaotic:
the trajectories that reach up-up form a measure-zero subset and the LQR
capture band is essentially never entered (see ``docs/NEXT_STEPS.md``,
2026-07-12 diagnostic: 0/15 captures). A swing-up that works must therefore
steer the *configuration*, not just the energy. This module implements the
standard remedy (Graichen, Treuer & Zeitz 2007; Tedrake, *Underactuated
Robotics*):

1. **Trajectory optimisation** (:func:`optimize_swingup`). Direct multiple
   shooting over the simulator's own discretisation. The horizon :math:`T` is
   split into :math:`N` segments of :math:`n_{\rm sub}` RK4 steps of the
   environment's :math:`dt`, with the force held constant on each segment.
   Decision variables are the knot states :math:`s_0 \dots s_N` and segment
   forces :math:`u_0 \dots u_{N-1}`; the defect constraints are

   .. math:: \Phi(s_k, u_k) - s_{k+1} = 0,

   where :math:`\Phi` is :math:`n_{\rm sub}` RK4 steps of the plant — exactly
   what :py:meth:`CartPendulumBase.step` integrates. Boundary conditions are
   hanging-at-rest :math:`\to` up-up-at-rest, with :math:`|x| \le x_{\max}`
   and :math:`|u| \le u_{\max}`; the objective is :math:`\sum_k h\,u_k^2`.
   Because the discretisation is the simulator's, the nominal trajectory is
   reproduced by the environment to solver tolerance (no transcription gap
   for the tracker to absorb).

2. **Time-varying LQR** (:func:`tvlqr_gains`). The RK4 step map is
   linearised along the nominal at every simulator step,
   :math:`\delta s_{j+1} \approx A_j\,\delta s_j + B_j\,\delta u_j`, and the
   discrete Riccati recursion

   .. math::

       K_j = (R + B_j^\top P_{j+1} B_j)^{-1} B_j^\top P_{j+1} A_j, \qquad
       P_j = Q + A_j^\top P_{j+1} (A_j - B_j K_j)

   is run backwards from :math:`P_{T}` = the infinite-horizon DARE solution at
   the upright equilibrium, so the tracking cost-to-go blends seamlessly into
   the balance controller's.

3. **Balance** — the discrete infinite-horizon LQR at up-up
   (:func:`upright_lqr_gain`).

:class:`SwingUpController` sequences them: an optional *settle* phase (LQR
about the hanging equilibrium, which removes the reset noise so the swing
starts from the nominal initial state), the tracked swing, then balance. It
outputs a force in newtons; wrap it with :class:`ForceControl` via
:py:meth:`SwingUpController.action` to drive an environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.optimize

UPRIGHT = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
HANGING = np.zeros(6)


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (or array of angles) to :math:`[-\\pi, \\pi)`."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def state_error(state: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``state - ref`` with the two pole angles wrapped to :math:`[-\\pi, \\pi)`."""
    err = np.asarray(state, dtype=np.float64) - ref
    err[..., 1:3] = wrap_angle(err[..., 1:3])
    return err


# --------------------------------------------------------------------------- #
# Plant
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlantParams:
    """Physical constants of the cart double pendulum (SI units)."""

    M: float = 1.0
    m1: float = 0.5
    m2: float = 0.5
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81
    friction_cart: float = 0.0
    friction_pole: float = 0.0
    dt: float = 0.005

    @classmethod
    def from_env(cls, env) -> PlantParams:
        """Snapshot the (curriculum-dependent) physics of a ``DoublePendulumCartEnv``."""
        return cls(
            M=float(env.M), m1=float(env.m1), m2=float(env.m2),
            l1=float(env.l1), l2=float(env.l2), g=float(env.g),
            friction_cart=float(env.friction_cart),
            friction_pole=float(env.friction_pole),
            dt=float(env.dt),
        )


def dynamics(states: np.ndarray, forces: np.ndarray, p: PlantParams) -> np.ndarray:
    r"""
    Batched :math:`\dot s = f(s, F)` for states of shape ``(..., 6)``.

    Same equations of motion as :py:meth:`DoublePendulumCartEnv._dynamics_into`
    (derived in ``docs/physics_derivation.md``), vectorised over leading axes.
    """
    x_dot, th1_dot, th2_dot = states[..., 3], states[..., 4], states[..., 5]
    th1, th2 = states[..., 1], states[..., 2]
    c1, s1 = np.cos(th1), np.sin(th1)
    c2, s2 = np.cos(th2), np.sin(th2)
    c12, s12 = np.cos(th1 - th2), np.sin(th1 - th2)
    m12 = p.m1 + p.m2

    shape = states.shape[:-1]
    Mm = np.empty(shape + (3, 3))
    Mm[..., 0, 0] = p.M + m12
    Mm[..., 0, 1] = Mm[..., 1, 0] = m12 * p.l1 * c1
    Mm[..., 0, 2] = Mm[..., 2, 0] = p.m2 * p.l2 * c2
    Mm[..., 1, 1] = m12 * p.l1 * p.l1
    Mm[..., 1, 2] = Mm[..., 2, 1] = p.m2 * p.l1 * p.l2 * c12
    Mm[..., 2, 2] = p.m2 * p.l2 * p.l2

    rhs = np.empty(shape + (3, 1))
    rhs[..., 0, 0] = (forces - p.friction_cart * x_dot
                      + m12 * p.l1 * s1 * th1_dot * th1_dot
                      + p.m2 * p.l2 * s2 * th2_dot * th2_dot)
    rhs[..., 1, 0] = (-p.friction_pole * th1_dot
                      - p.m2 * p.l1 * p.l2 * s12 * th2_dot * th2_dot
                      - m12 * p.g * p.l1 * s1)
    rhs[..., 2, 0] = (-p.friction_pole * th2_dot
                      + p.m2 * p.l1 * p.l2 * s12 * th1_dot * th1_dot
                      - p.m2 * p.g * p.l2 * s2)
    q_dd = np.linalg.solve(Mm, rhs)[..., 0]
    return np.concatenate([states[..., 3:], q_dd], axis=-1)


def rk4_step(states: np.ndarray, forces: np.ndarray, p: PlantParams, dt: float | None = None
             ) -> np.ndarray:
    """One classical RK4 step (the environment's default integrator), batched."""
    h = p.dt if dt is None else dt
    k1 = dynamics(states, forces, p)
    k2 = dynamics(states + 0.5 * h * k1, forces, p)
    k3 = dynamics(states + 0.5 * h * k2, forces, p)
    k4 = dynamics(states + h * k3, forces, p)
    return states + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def linearize_step(states: np.ndarray, forces: np.ndarray, p: PlantParams,
                   eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Central-difference Jacobians of the RK4 step map at a batch of points.

    Returns ``A`` of shape ``(n, 6, 6)`` and ``B`` of shape ``(n, 6, 1)`` with
    :math:`A = \partial \Phi / \partial s`, :math:`B = \partial \Phi / \partial F`.
    All :math:`2 \cdot 7` perturbations of every point go through one batched
    RK4 call.
    """
    states = np.atleast_2d(states)
    forces = np.atleast_1d(forces).astype(np.float64)
    pert = np.zeros((14, 7))
    pert[np.arange(7), np.arange(7)] = eps
    pert[7 + np.arange(7), np.arange(7)] = -eps
    s = states[:, None, :] + pert[None, :, :6]
    u = forces[:, None] + pert[None, :, 6]
    out = rk4_step(s, u, p)
    jac = (out[:, :7] - out[:, 7:]) / (2.0 * eps)  # (n, 7, 6): d out / d input_i
    A = np.transpose(jac[:, :6, :], (0, 2, 1))
    B = jac[:, 6, :][:, :, None]
    return A, B


def lqr_gain_at(p: PlantParams, eq: np.ndarray, Q: np.ndarray | None = None, R: float = 0.01
                ) -> tuple[np.ndarray, np.ndarray]:
    """Discrete infinite-horizon LQR about the equilibrium ``eq`` (zero force);
    returns ``(K, P)`` with ``u = -K e``."""
    Q = default_q() if Q is None else Q
    A, B = linearize_step(np.asarray(eq, dtype=np.float64), 0.0, p)
    return _dlqr(A[0], B[0], Q, np.array([[R]]))


def upright_lqr_gain(p: PlantParams, Q: np.ndarray | None = None, R: float = 0.01
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Discrete infinite-horizon LQR at up-up; returns ``(K, P)`` with ``u = -K e``."""
    return lqr_gain_at(p, UPRIGHT, Q, R)


def hanging_lqr_gain(p: PlantParams, Q: np.ndarray | None = None, R: float = 0.01
                     ) -> np.ndarray:
    """Discrete infinite-horizon LQR about the hanging equilibrium (settle phase)."""
    return lqr_gain_at(p, HANGING, Q, R)[0]


def default_q() -> np.ndarray:
    return np.diag([10.0, 100.0, 100.0, 1.0, 1.0, 1.0])


def _dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# --------------------------------------------------------------------------- #
# Trajectory optimisation
# --------------------------------------------------------------------------- #


@dataclass
class SwingUpTrajectory:
    """
    Nominal swing-up on the simulator grid.

    ``states[j]`` is the nominal state at simulator step ``j`` (``len = n + 1``)
    and ``forces[j]`` the feed-forward force applied during step ``j``
    (``len = n``). ``goal`` is the terminal equilibrium actually reached (its
    pole angles are :math:`\\pm\\pi`; the sign records which way each pole
    swung).
    """

    states: np.ndarray
    forces: np.ndarray
    goal: np.ndarray
    params: PlantParams
    info: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.forces) * self.params.dt

    def save(self, path: str | Path) -> None:
        p = self.params
        np.savez(
            path, states=self.states, forces=self.forces, goal=self.goal,
            params=np.array([p.M, p.m1, p.m2, p.l1, p.l2, p.g,
                             p.friction_cart, p.friction_pole, p.dt]),
        )

    @classmethod
    def load(cls, path: str | Path) -> SwingUpTrajectory:
        d = np.load(path)
        return cls(states=d["states"], forces=d["forces"], goal=d["goal"],
                   params=PlantParams(*map(float, d["params"])))


def _segment_rollout(s: np.ndarray, u: np.ndarray, p: PlantParams, n_sub: int) -> np.ndarray:
    for _ in range(n_sub):
        s = rk4_step(s, u, p)
    return s


def optimize_swingup(
    p: PlantParams,
    duration: float = 3.0,
    n_sub: int = 10,
    u_max: float = 150.0,
    x_max: float = 2.5,
    goal: np.ndarray | None = None,
    start: np.ndarray | None = None,
    maxiter: int = 400,
    seed: int = 0,
    verbose: bool = False,
) -> SwingUpTrajectory:
    r"""
    Solve the hanging :math:`\to` up-up swing-up by direct multiple shooting.

    Parameters
    ----------
    p : physics (use :py:meth:`PlantParams.from_env`).
    duration : swing horizon :math:`T` [s]; rounded to whole segments.
    n_sub : simulator steps per segment (force held constant over a segment).
    u_max : force bound [N] on the nominal (leave headroom for feedback).
    x_max : cart-position bound [m] on the nominal.
    goal : terminal state; default :data:`UPRIGHT`. Pole angles of
        :math:`\pm\pi` select the swing direction of each link.
    start : initial state; default :data:`HANGING`.
    seed : seeds the small random perturbation of the initial guess.

    Returns
    -------
    :class:`SwingUpTrajectory` resampled onto the simulator grid. ``info``
    holds the solver status, cost, and maximum defect.
    """
    goal = UPRIGHT.copy() if goal is None else np.asarray(goal, dtype=np.float64)
    start = HANGING.copy() if start is None else np.asarray(start, dtype=np.float64)
    h = n_sub * p.dt
    N = max(2, int(round(duration / h)))
    n_s = 6 * (N + 1)

    def unpack(z):
        return z[:n_s].reshape(N + 1, 6), z[n_s:]

    # Initial guess: smooth (cosine) interpolation of the angles, zero cart
    # motion, velocities consistent with the interpolation.
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, 1.0, N + 1)
    blend = 0.5 - 0.5 * np.cos(np.pi * tau)
    dblend = 0.5 * np.pi * np.sin(np.pi * tau) / (N * h)
    S0 = start[None, :] + blend[:, None] * (goal - start)[None, :]
    S0[:, 4:6] = dblend[:, None] * (goal - start)[None, 1:3]
    S0[1:-1] += 0.01 * rng.standard_normal((N - 1, 6))
    z0 = np.concatenate([S0.ravel(), np.zeros(N)])

    u_scale = u_max

    def cost(z):
        _, u = unpack(z)
        return h * float(u @ u) / (u_scale * u_scale)

    def cost_grad(z):
        _, u = unpack(z)
        g = np.zeros_like(z)
        g[n_s:] = 2.0 * h * u / (u_scale * u_scale)
        return g

    def defects(z):
        S, u = unpack(z)
        nxt = _segment_rollout(S[:-1], u, p, n_sub)
        d = nxt - S[1:]
        return np.concatenate([S[0] - start, d.ravel(), state_error(S[-1], goal)])

    def defects_jac(z):
        S, u = unpack(z)
        # Chain the per-step Jacobians across each segment.
        s = S[:-1].copy()
        Phi_s = np.broadcast_to(np.eye(6), (N, 6, 6)).copy()
        Phi_u = np.zeros((N, 6, 1))
        for _ in range(n_sub):
            A, B = linearize_step(s, u, p)
            Phi_s = A @ Phi_s
            Phi_u = A @ Phi_u + B
            s = rk4_step(s, u, p)
        J = np.zeros((6 * (N + 2), z.size))
        J[:6, :6] = np.eye(6)
        for k in range(N):
            r = 6 + 6 * k
            J[r:r + 6, 6 * k:6 * k + 6] = Phi_s[k]
            J[r:r + 6, 6 * (k + 1):6 * (k + 1) + 6] = -np.eye(6)
            J[r:r + 6, n_s + k] = Phi_u[k, :, 0]
        J[-6:, 6 * N:6 * N + 6] = np.eye(6)
        return J

    bounds = [(None, None)] * n_s + [(-u_max, u_max)] * N
    for k in range(N + 1):
        bounds[6 * k] = (-x_max, x_max)

    res = scipy.optimize.minimize(
        cost, z0, jac=cost_grad, method="SLSQP", bounds=bounds,
        constraints=[{"type": "eq", "fun": defects, "jac": defects_jac}],
        options={"maxiter": maxiter, "ftol": 1e-9, "disp": verbose},
    )
    # SLSQP gets close to feasible long before it certifies optimality; finish
    # with minimum-norm Gauss-Newton steps on the defects alone.
    z = res.x.copy()
    for _ in range(20):
        c = defects(z)
        if np.max(np.abs(c)) < 1e-10:
            break
        z -= np.linalg.lstsq(defects_jac(z), c, rcond=None)[0]
    S, u = unpack(z)
    max_defect = float(np.max(np.abs(defects(z))))

    # Resample onto the simulator grid by integrating each segment from its
    # knot (not open-loop from s_0, which would amplify the tiny defects).
    fine_u = np.repeat(u, n_sub)
    fine_s = np.empty((N * n_sub + 1, 6))
    s = S[:-1].copy()
    for i in range(n_sub):
        fine_s[i:N * n_sub:n_sub] = s
        s = rk4_step(s, u, p)
    fine_s[-1] = S[-1]
    info = {"success": bool(res.success), "message": str(res.message),
            "cost": cost(z), "max_defect": max_defect, "nit": int(res.nit),
            "segments": N, "n_sub": n_sub, "u_max": u_max, "x_max": x_max}
    return SwingUpTrajectory(states=fine_s, forces=fine_u, goal=goal, params=p, info=info)


def is_feasible(traj: SwingUpTrajectory, tol: float = 1e-6) -> bool:
    """Dynamically consistent and (to a small tolerance) within its bounds."""
    info = traj.info
    return (info["max_defect"] < tol
            and np.abs(traj.states[:, 0]).max() <= info["x_max"] + 0.05
            and np.abs(traj.forces).max() <= info["u_max"] + 1e-6)


def best_swingup(p: PlantParams, durations=(4.0, 4.5, 3.5, 5.0), verbose: bool = False,
                 **kwargs) -> SwingUpTrajectory:
    """
    Try :func:`optimize_swingup` over the given horizons and both relative
    swing directions (poles to :math:`(\\pi, \\pi)`, then :math:`(\\pi, -\\pi)`)
    and return the first feasible solution. ``T = 4`` s with :math:`(\\pi, \\pi)`
    solves at :math:`\\delta = 1`; the rest are fallbacks for other physics.
    """
    for T in durations:
        for sign2 in (1.0, -1.0):
            goal = UPRIGHT.copy()
            goal[2] *= sign2
            traj = optimize_swingup(p, duration=T, goal=goal, **kwargs)
            ok = is_feasible(traj)
            if verbose:
                print(f"T={T:.2f} goal=({goal[1]:+.2f},{goal[2]:+.2f}) feasible={ok} "
                      f"cost={traj.info['cost']:.4f} nit={traj.info['nit']} "
                      f"defect={traj.info['max_defect']:.1e}")
            if ok:
                return traj
    raise RuntimeError("No feasible swing-up trajectory found; try longer durations "
                       "or a larger u_max.")


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #


def tvlqr_gains(traj: SwingUpTrajectory, Q: np.ndarray | None = None, R: float = 0.01,
                P_final: np.ndarray | None = None) -> np.ndarray:
    """
    Finite-horizon discrete LQR gains along ``traj``; returns ``K`` of shape
    ``(n, 1, 6)`` with feedback ``u_j = F_j - K_j (s_j - s^{nom}_j)``. The
    terminal cost defaults to the infinite-horizon LQR at ``traj.goal``.
    """
    p = traj.params
    Q = default_q() if Q is None else Q
    Rm = np.array([[R]])
    if P_final is None:
        P_final = lqr_gain_at(p, traj.goal, Q, R)[1]
    A, B = linearize_step(traj.states[:-1], traj.forces, p)
    n = len(traj.forces)
    K = np.empty((n, 1, 6))
    P = P_final
    for j in range(n - 1, -1, -1):
        Aj, Bj = A[j], B[j]
        BtP = Bj.T @ P
        K[j] = np.linalg.solve(Rm + BtP @ Bj, BtP @ Aj)
        P = Q + Aj.T @ P @ (Aj - Bj @ K[j])
        P = 0.5 * (P + P.T)
    return K


class SwingUpController:
    r"""
    Settle :math:`\to` tracked swing-up :math:`\to` LQR balance.

    Call :py:meth:`reset` at the start of every episode and then
    :py:meth:`force` (newtons) or :py:meth:`action` (normalised for a
    :class:`ForceControl` with ``max_force``) once per environment step.

    Phases
    ------
    ``settle``  LQR about hanging for ``settle_time`` seconds, or until the
                state is within ``settle_tol`` of rest (whichever is first
                after ``min_settle_time``). Removes the reset noise so the
                swing starts on its nominal initial state.
    ``swing``   :math:`u = F_j - K_j (s - s^{\rm nom}_j)` along the trajectory.
    ``balance`` :math:`u = -K_\infty (s - s^\star)` at up-up.

    If the state is already inside the balance band on reset (e.g. an ``up``
    reset), the controller goes straight to ``balance``.
    """

    def __init__(
        self,
        traj: SwingUpTrajectory,
        Q: np.ndarray | None = None,
        R: float = 0.01,
        force_limit: float = 500.0,
        settle_time: float = 3.0,
        min_settle_time: float = 0.5,
        settle_tol: float = 0.02,
        balance_band: float = 0.35,
    ) -> None:
        self.traj = traj
        p = traj.params
        self.params = p
        self.K_balance, P_up = upright_lqr_gain(p, Q, R)
        self.K_track = tvlqr_gains(traj, Q, R, P_final=P_up)
        self.K_settle = hanging_lqr_gain(p, Q, R)
        self.goal = traj.goal
        self.start = traj.states[0]
        self.force_limit = float(force_limit)
        self.settle_steps = int(round(settle_time / p.dt))
        self.min_settle_steps = int(round(min_settle_time / p.dt))
        self.settle_tol = float(settle_tol)
        self.balance_band = float(balance_band)
        self.reset()

    def reset(self, state: np.ndarray | None = None) -> None:
        self.phase = "settle"
        self._k = 0
        if state is not None and np.all(np.abs(wrap_angle(np.asarray(state)[1:3] - np.pi))
                                        < self.balance_band):
            self.phase = "balance"

    def force(self, state: np.ndarray) -> float:
        s = np.asarray(state, dtype=np.float64)
        if self.phase == "settle":
            err = state_error(s, self.start)
            done = (self._k >= self.settle_steps
                    or (self._k >= self.min_settle_steps
                        and np.max(np.abs(err)) < self.settle_tol))
            if done:
                self.phase, self._k = "swing", 0
            else:
                self._k += 1
                return self._clip(float(-(self.K_settle @ err)[0]))
        if self.phase == "swing":
            if self._k >= len(self.traj.forces):
                self.phase = "balance"
            else:
                j = self._k
                self._k += 1
                err = state_error(s, self.traj.states[j])
                return self._clip(float(self.traj.forces[j] - (self.K_track[j] @ err)[0]))
        err = state_error(s, self.goal)
        return self._clip(float(-(self.K_balance @ err)[0]))

    def action(self, state: np.ndarray, max_force: float = 5000.0) -> np.ndarray:
        """Normalised action for :class:`ForceControl` (``F = a * max_force``)."""
        return np.array([self.force(state) / max_force], dtype=np.float32)

    def _clip(self, f: float) -> float:
        return float(np.clip(f, -self.force_limit, self.force_limit))
