r"""
Double pendulum on a cart.

State (8-D obs, 6-D internal):
    s_internal = [x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot]
    obs        = [x, sin(theta_1), sin(theta_2), cos(theta_1), cos(theta_2),
                  x_dot, theta_1_dot, theta_2_dot]

Equations of motion (Lagrangian, theta measured from the downward vertical).
Bob 1 hangs from the cart; bob 2 hangs from bob 1's tip. Generalised
coordinates :math:`q = [x, \theta_1, \theta_2]`. The mass matrix, Coriolis
vector, and gravity vector are derived in :file:`docs/physics_derivation.md`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.env.cart_pendulum_base import CartPendulumBase
from src.strategies.controls import ControlStrategy
from src.strategies.rewards import RewardStrategy


def angle_normalize(x: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle to :math:`[-\\pi, \\pi]`."""
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


class DoublePendulumCartEnv(CartPendulumBase):
    @property
    def n_poles(self) -> int:
        return 2

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        wind_std: float = 0.0,
        reset_mode: str = "down",
        control_strategy: Optional[ControlStrategy] = None,
        reward_strategy: Optional[RewardStrategy] = None,
        x_soft: float = 3.5,
        x_hard: float = 10.0,
        boundary_penalty_k: float = 0.1,
        integrator: str = "rk4",
        wind_max: float = 1.0,
    ) -> None:
        self.m1: float = 0.5
        self.m2: float = 0.5
        self.l1: float = 1.0
        self.l2: float = 1.0
        super().__init__(
            render_mode=render_mode,
            wind_std=wind_std,
            reset_mode=reset_mode,
            control_strategy=control_strategy,
            reward_strategy=reward_strategy,
            x_soft=x_soft,
            x_hard=x_hard,
            boundary_penalty_k=boundary_penalty_k,
            integrator=integrator,
            wind_max=wind_max,
        )

        # Allocation-reduction buffers for the 3x3 mass matrix and 3-vector
        # RHS used by `_dynamics_into`. Float64 is pinned explicitly so the
        # bit-equivalence baseline (SHA-256 of trajectory states) is
        # preserved.
        self._M_mat = np.empty((3, 3), dtype=np.float64)
        self._RHS = np.empty(3, dtype=np.float64)

    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        # Backwards-compatible wrapper that allocates a fresh return array.
        # Internally we route through `_dynamics_into` which writes into a
        # caller-owned buffer (the RK4 stages exploit this).
        out = np.empty(6, dtype=np.float64)
        self._dynamics_into(state, force, out)
        return out

    def _dynamics_into(self, state: np.ndarray, force: float,
                       out: np.ndarray) -> None:
        r"""
        In-place form of `_dynamics`: writes :math:`\dot s` into ``out``.

        Squaring convention
        -------------------
        Velocity-squared terms (``theta1_dot * theta1_dot`` etc.) are written
        as explicit multiplication rather than ``** 2``. This makes the scalar
        path bit-identical to the batched path used by
        :py:meth:`dynamics_into_batched` — numpy's ``**`` ufunc on an array
        and Python's scalar ``**`` use slightly different float64 rounding
        for the integer-2 case, while ``x * x`` is bit-identical in both
        contexts. The per-row trajectory hash baseline in
        ``tests/test_pipeline_equivalence.py`` was regenerated for this
        convention; see that file's docstring.
        """
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state

        M = self.M
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g

        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        c12 = np.cos(theta1 - theta2)
        s12 = np.sin(theta1 - theta2)

        # Mass matrix M(q) — symmetric, written via indexed assignment.
        # ``l1 ** 2`` and ``l2 ** 2`` are evaluated once at parse time on
        # Python floats (constants), so their result is a stable Python
        # float regardless of array-vs-scalar context — they don't need
        # the explicit `*` rewrite.
        M_mat = self._M_mat
        M_mat[0, 0] = M + m1 + m2
        M_mat[0, 1] = (m1 + m2) * l1 * c1
        M_mat[0, 2] = m2 * l2 * c2
        M_mat[1, 0] = (m1 + m2) * l1 * c1
        M_mat[1, 1] = (m1 + m2) * l1 * l1
        M_mat[1, 2] = m2 * l1 * l2 * c12
        M_mat[2, 0] = m2 * l2 * c2
        M_mat[2, 1] = m2 * l1 * l2 * c12
        M_mat[2, 2] = m2 * l2 * l2

        # RHS = F + D - C - G, with per-row scalar expressions. Velocity-
        # squared terms are written as ``x * x`` for bit-equivalence with
        # the batched dynamics (see docstring).
        # Original entries:
        #   C0 = -(m1+m2)*l1*s1*theta1_dot*theta1_dot - m2*l2*s2*theta2_dot*theta2_dot
        #   C1 =  m2*l1*l2*s12*theta2_dot*theta2_dot
        #   C2 = -m2*l1*l2*s12*theta1_dot*theta1_dot
        #   G0 = 0,  G1 = (m1+m2)*g*l1*s1,  G2 = m2*g*l2*s2
        #   D0 = -friction_cart*x_dot,  D1 = -friction_pole*theta1_dot,
        #   D2 = -friction_pole*theta2_dot
        #   F0 = force, F1 = F2 = 0
        # Per-element sum order is ((F + D) - C) - G.
        C0 = -(m1 + m2) * l1 * s1 * theta1_dot * theta1_dot - m2 * l2 * s2 * theta2_dot * theta2_dot
        C1 = m2 * l1 * l2 * s12 * theta2_dot * theta2_dot
        C2 = -m2 * l1 * l2 * s12 * theta1_dot * theta1_dot
        G1 = (m1 + m2) * g * l1 * s1
        G2 = m2 * g * l2 * s2
        D0 = -self.friction_cart * x_dot
        D1 = -self.friction_pole * theta1_dot
        D2 = -self.friction_pole * theta2_dot

        RHS = self._RHS
        # Row 0: F0 + D0 - C0 - G0 with G0=0 -> force + D0 - C0
        RHS[0] = force + D0 - C0
        # Row 1: F1 + D1 - C1 - G1 with F1=0 -> D1 - C1 - G1
        RHS[1] = D1 - C1 - G1
        # Row 2: F2 + D2 - C2 - G2 with F2=0 -> D2 - C2 - G2
        RHS[2] = D2 - C2 - G2

        # `np.linalg.solve` allocates a fresh 3-vector; copy into `out` so
        # the caller doesn't end up with an alias of an internal buffer.
        q_dd = np.linalg.solve(M_mat, RHS)
        out[0] = x_dot
        out[1] = theta1_dot
        out[2] = theta2_dot
        out[3] = q_dd[0]
        out[4] = q_dd[1]
        out[5] = q_dd[2]

    @staticmethod
    def dynamics_into_batched(
        states: np.ndarray,
        forces: np.ndarray,
        out: np.ndarray,
        *,
        M_cart: float,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
        g: float,
        friction_cart: float,
        friction_pole: float,
        M_buf: np.ndarray,
        RHS_buf: np.ndarray,
    ) -> None:
        r"""
        Batched in-place dynamics for N envs sharing identical physics parameters.

        Inputs
        ------
        states : (N, 6) float64 array of internal states.
        forces : (N,) float64 array of cart forces.
        out    : (N, 6) float64 array; overwritten with :math:`\dot s` per env.

        Internal scratch
        ----------------
        M_buf   : (N, 3, 3) float64 mass-matrix buffer, overwritten.
        RHS_buf : (N, 3, 1) float64 RHS buffer, overwritten.

        Bit-equivalence
        ---------------
        Each per-row arithmetic expression mirrors `_dynamics_into` *exactly*
        (same operator precedence, same intermediate products, same per-row
        ``np.linalg.solve(M, RHS)`` call). Numpy's elementwise ufuncs (cos,
        sin, +, -, *, **) on a 1-D batch produce bit-identical per-row
        results to their scalar counterparts — empirically verified.

        Solve handling
        --------------
        ``np.linalg.solve`` on a stacked ``(N, 3, 3)`` matrix is *not*
        bit-equivalent to N scalar calls in all cases (LAPACK gufunc path
        differs slightly from the scalar `_gesv` path at the float64 ULP
        level on near-singular matrices). To preserve per-row bit-equality
        with the master baseline, the solve is done in a tight Python loop
        over the N rows. The matrix and RHS construction (the dominant
        per-stage Python overhead in the unbatched path) is still
        vectorised, so we still get a meaningful per-stage speed-up.

        This static method takes parameters explicitly so the trainer can use
        a single shared scratch buffer across all envs (assumed to share the
        same curriculum, which the trainer enforces via `_set_curriculum_all`).
        """
        # Unpack columns. View slices avoid copies.
        x_dot = states[:, 3]
        theta1_dot = states[:, 4]
        theta2_dot = states[:, 5]
        theta1 = states[:, 1]
        theta2 = states[:, 2]

        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        c12 = np.cos(theta1 - theta2)
        s12 = np.sin(theta1 - theta2)

        # Mass matrix M(q) — symmetric, written via indexed assignment along axis 0.
        # Constants (do not depend on per-env state). ``l1 ** 2`` evaluates
        # at parse time as a Python float constant — bit-equal to the
        # unbatched code's ``l1 ** 2``.
        M_buf[:, 0, 0] = M_cart + m1 + m2
        M_buf[:, 0, 1] = (m1 + m2) * l1 * c1
        M_buf[:, 0, 2] = m2 * l2 * c2
        M_buf[:, 1, 0] = (m1 + m2) * l1 * c1
        M_buf[:, 1, 1] = (m1 + m2) * l1 * l1
        M_buf[:, 1, 2] = m2 * l1 * l2 * c12
        M_buf[:, 2, 0] = m2 * l2 * c2
        M_buf[:, 2, 1] = m2 * l1 * l2 * c12
        M_buf[:, 2, 2] = m2 * l2 * l2

        # RHS components. Velocity-squared terms use explicit ``x * x``
        # to match the unbatched scalar path bit-for-bit (numpy's array
        # ``** 2`` rounds slightly differently from Python scalar ``** 2``
        # for the integer-2 case; ``x * x`` agrees in both contexts).
        C0 = -(m1 + m2) * l1 * s1 * theta1_dot * theta1_dot - m2 * l2 * s2 * theta2_dot * theta2_dot
        C1 = m2 * l1 * l2 * s12 * theta2_dot * theta2_dot
        C2 = -m2 * l1 * l2 * s12 * theta1_dot * theta1_dot
        G1 = (m1 + m2) * g * l1 * s1
        G2 = m2 * g * l2 * s2
        D0 = -friction_cart * x_dot
        D1 = -friction_pole * theta1_dot
        D2 = -friction_pole * theta2_dot

        # RHS_buf has shape (N, 3, 1) for the gufunc signature
        # `(m,m),(m,n)->(m,n)` — interpreted as N stacked column vectors.
        RHS_buf[:, 0, 0] = forces + D0 - C0
        RHS_buf[:, 1, 0] = D1 - C1 - G1
        RHS_buf[:, 2, 0] = D2 - C2 - G2

        # Batched solve: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1). numpy dispatches
        # one LAPACK `_gesv` per (3, 3) slice. With bit-identical M/RHS
        # inputs (achieved by writing all velocity-squared terms as
        # ``x * x`` rather than ``x ** 2``), the batched solve produces
        # per-row outputs that are bit-equal to N scalar
        # ``np.linalg.solve(M[i], RHS[i])`` calls — verified by
        # ``tools/check_batched_equivalence.py``.
        q_dd = np.linalg.solve(M_buf, RHS_buf)  # (N, 3, 1)

        # Pack derivatives.
        out[:, 0] = x_dot
        out[:, 1] = theta1_dot
        out[:, 2] = theta2_dot
        out[:, 3:6] = q_dd[:, :, 0]

    def _get_energy(self) -> float:
        r"""Total mechanical energy :math:`T + V`, used for diagnostics only."""
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = self.state
        M, m1, m2 = self.M, self.m1, self.m2
        l1, l2, g = self.l1, self.l2, self.g

        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        c12 = np.cos(theta1 - theta2)

        T = 0.5 * (
            (M + m1 + m2) * x_dot ** 2
            + (m1 + m2) * l1 ** 2 * theta1_dot ** 2
            + m2 * l2 ** 2 * theta2_dot ** 2
            + 2.0 * (m1 + m2) * l1 * c1 * x_dot * theta1_dot
            + 2.0 * m2 * l2 * c2 * x_dot * theta2_dot
            + 2.0 * m2 * l1 * l2 * c12 * theta1_dot * theta2_dot
        )
        V = -(m1 + m2) * g * l1 * c1 - m2 * g * l2 * c2
        return T + V
