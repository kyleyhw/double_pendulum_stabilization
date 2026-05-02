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

        Bit-equivalence
        ---------------
        Each scalar arithmetic expression is preserved verbatim from the
        legacy `_dynamics` body — same operator precedence, same intermediate
        products, same `np.linalg.solve(M, RHS)` call. Only the *allocation*
        pattern changes (indexed writes into pre-allocated `_M_mat`/`_RHS`
        buffers instead of `np.array([...])` constructors). The trajectory
        SHA-256 hash captured on master is therefore preserved.
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
        M_mat = self._M_mat
        M_mat[0, 0] = M + m1 + m2
        M_mat[0, 1] = (m1 + m2) * l1 * c1
        M_mat[0, 2] = m2 * l2 * c2
        M_mat[1, 0] = (m1 + m2) * l1 * c1
        M_mat[1, 1] = (m1 + m2) * l1 ** 2
        M_mat[1, 2] = m2 * l1 * l2 * c12
        M_mat[2, 0] = m2 * l2 * c2
        M_mat[2, 1] = m2 * l1 * l2 * c12
        M_mat[2, 2] = m2 * l2 ** 2

        # RHS = F + D - C - G, with the original per-row scalar expressions
        # preserved verbatim. The original code computed C_vec, G_vec, D_vec,
        # F_vec separately and summed via numpy `+` / `-`; here we fuse the
        # arithmetic per row to avoid the four temporary arrays.
        # Original entries:
        #   C0 = -(m1+m2)*l1*s1*theta1_dot**2 - m2*l2*s2*theta2_dot**2
        #   C1 =  m2*l1*l2*s12*theta2_dot**2
        #   C2 = -m2*l1*l2*s12*theta1_dot**2
        #   G0 = 0,  G1 = (m1+m2)*g*l1*s1,  G2 = m2*g*l2*s2
        #   D0 = -friction_cart*x_dot,  D1 = -friction_pole*theta1_dot,
        #   D2 = -friction_pole*theta2_dot
        #   F0 = force, F1 = F2 = 0
        # Sum-order: numpy evaluates `F_vec + D_vec - C_vec - G_vec`
        # left-to-right so the per-element sum is `((F + D) - C) - G`. We
        # mirror that ordering here so the float64 sums match bitwise.
        C0 = -(m1 + m2) * l1 * s1 * theta1_dot ** 2 - m2 * l2 * s2 * theta2_dot ** 2
        C1 = m2 * l1 * l2 * s12 * theta2_dot ** 2
        C2 = -m2 * l1 * l2 * s12 * theta1_dot ** 2
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
