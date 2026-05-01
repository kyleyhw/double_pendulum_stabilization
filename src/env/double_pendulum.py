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

    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state

        M, m1, m2 = self.M, self.m1, self.m2
        l1, l2, g = self.l1, self.l2, self.g

        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c12 = np.cos(theta1 - theta2)
        s12 = np.sin(theta1 - theta2)

        M_mat = np.array(
            [
                [M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2],
                [(m1 + m2) * l1 * c1, (m1 + m2) * l1 ** 2, m2 * l1 * l2 * c12],
                [m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2 ** 2],
            ]
        )

        # Coriolis / centrifugal vector (already in the form C(q, qdot) qdot).
        C_vec = np.array(
            [
                -(m1 + m2) * l1 * s1 * theta1_dot ** 2 - m2 * l2 * s2 * theta2_dot ** 2,
                m2 * l1 * l2 * s12 * theta2_dot ** 2,
                -m2 * l1 * l2 * s12 * theta1_dot ** 2,
            ]
        )

        # G(q) = dV/dq with V = -(m1+m2) g l1 cos(theta1) - m2 g l2 cos(theta2):
        #   dV/dtheta1 = +(m1 + m2) g l1 sin(theta1)
        #   dV/dtheta2 = +m2 g l2 sin(theta2)
        G_vec = np.array(
            [
                0.0,
                (m1 + m2) * g * l1 * s1,
                m2 * g * l2 * s2,
            ]
        )

        D_vec = np.array(
            [
                -self.friction_cart * x_dot,
                -self.friction_pole * theta1_dot,
                -self.friction_pole * theta2_dot,
            ]
        )
        F_vec = np.array([force, 0.0, 0.0])

        # M qdd + C + G = F + D  =>  qdd = M^{-1}(F + D - C - G)
        q_dd = np.linalg.solve(M_mat, F_vec + D_vec - C_vec - G_vec)
        return np.concatenate([[x_dot, theta1_dot, theta2_dot], q_dd])

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
