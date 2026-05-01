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

    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        x, theta, x_dot, theta_dot = state
        M, m, l, g = self.M, self.m, self.l, self.g

        c, s = np.cos(theta), np.sin(theta)
        M_mat = np.array([[M + m, m * l * c], [m * l * c, m * l ** 2]])
        # Coriolis vector C(q, qdot) (vector form of C(q, qdot) qdot for this 2-DoF system).
        C_vec = np.array([-m * l * s * theta_dot ** 2, 0.0])
        # Gravity G(q) = dV/dq with V = -m g l cos(theta), giving G = [0, m g l sin(theta)].
        G_vec = np.array([0.0, m * g * l * s])

        D_vec = np.array([-self.friction_cart * x_dot, -self.friction_pole * theta_dot])
        F_vec = np.array([force, 0.0])

        # M qdd + C + G = F + D  =>  qdd = M^{-1}(F + D - C - G)
        q_dd = np.linalg.solve(M_mat, F_vec + D_vec - C_vec - G_vec)
        return np.concatenate([[x_dot, theta_dot], q_dd])
