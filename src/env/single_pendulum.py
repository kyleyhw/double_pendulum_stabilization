r"""
Single pendulum on a cart.

State (5-D obs, 4-D internal):
    s_internal = [x, theta, x_dot, theta_dot]
    obs        = [x, sin(theta), cos(theta), x_dot, theta_dot]

Equations of motion (Lagrangian):
    M(q) qdd + C(q, qdot) + G(q) = F
where q = [x, theta] and

    M(q) = [[M+m, m l cos(theta)],
            [m l cos(theta), m l^2]]

    C(q, qdot) = [-m l sin(theta) theta_dot^2, 0]
    G(q)       = [0, m g l sin(theta)]
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.env.cart_pendulum_base import CartPendulumBase
from src.strategies.controls import ControlStrategy
from src.strategies.rewards import RewardStrategy


class SinglePendulumCartEnv(CartPendulumBase):
    @property
    def n_poles(self) -> int:
        return 1

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
        # Pole physics: must be set BEFORE super().__init__ because the base
        # uses self.n_poles in _build_observation_space, but the pole *masses*
        # are not needed there. We set them here for clarity.
        self.m: float = 0.5
        self.l: float = 1.0
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
        # Convenience aliases used by the visualiser.
        self.l1 = self.l

        # Allocation-reduction buffers for the 2x2 mass matrix and 2-vector
        # RHS used by `_dynamics_into`. Float64 pinned to preserve the
        # bit-equivalence baseline.
        self._M_mat = np.empty((2, 2), dtype=np.float64)
        self._RHS = np.empty(2, dtype=np.float64)

    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        # Backwards-compat wrapper that allocates a fresh return array.
        out = np.empty(4, dtype=np.float64)
        self._dynamics_into(state, force, out)
        return out

    def _dynamics_into(self, state: np.ndarray, force: float,
                       out: np.ndarray) -> None:
        r"""In-place form of `_dynamics`. Bit-equivalent to the original.

        See `double_pendulum.DoublePendulumCartEnv._dynamics_into` for the
        general approach. Per-row scalar arithmetic is preserved verbatim.
        """
        x, theta, x_dot, theta_dot = state
        M = self.M
        m = self.m
        l = self.l
        g = self.g

        c = np.cos(theta)
        s = np.sin(theta)

        M_mat = self._M_mat
        M_mat[0, 0] = M + m
        M_mat[0, 1] = m * l * c
        M_mat[1, 0] = m * l * c
        M_mat[1, 1] = m * l ** 2

        # Original entries (preserved):
        #   C = [-m*l*s*theta_dot**2, 0],  G = [0, m*g*l*s]
        #   D = [-friction_cart*x_dot, -friction_pole*theta_dot]
        #   F = [force, 0]
        # Per-row RHS = F + D - C - G.
        C0 = -m * l * s * theta_dot ** 2
        G1 = m * g * l * s
        D0 = -self.friction_cart * x_dot
        D1 = -self.friction_pole * theta_dot

        RHS = self._RHS
        RHS[0] = force + D0 - C0  # G0 = 0
        RHS[1] = D1 - G1          # F1 = 0, C1 = 0

        q_dd = np.linalg.solve(M_mat, RHS)
        out[0] = x_dot
        out[1] = theta_dot
        out[2] = q_dd[0]
        out[3] = q_dd[1]
