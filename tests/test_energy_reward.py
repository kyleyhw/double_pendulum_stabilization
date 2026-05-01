r"""
Tests for :class:`EnergyShapingReward` (the convex combination of an energy
proximity term, an angle-error term, and a gated kinetic damping term).

These tests fix the body parameters to the values used by the env defaults
(M = 1, m_i = 0.5, l_i = 1, g = 9.81) and probe the reward at canonical states.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402
from src.strategies.rewards import EnergyShapingReward  # noqa: E402


class TestEnergyShapingReward(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DoublePendulumCartEnv(
            control_strategy=ForceControl(),
            reward_strategy=EnergyShapingReward(),
        )
        self.env.set_curriculum(1.0)  # tightest tolerance

    def _r(self, state: np.ndarray) -> float:
        self.env.reset(seed=0)
        self.env.state = state.astype(np.float64)
        _, r, _, _, _ = self.env.step(np.zeros(1, dtype=np.float32))
        return float(r)

    def test_upright_rest_attains_max(self):
        """At upright with zero velocity all three components are at maximum."""
        r = self._r(np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
        self.assertGreater(r, 0.99)

    def test_down_rest_lower_than_upright(self):
        """Down-rest has a large energy gap and angle error -> reward smaller."""
        r_up = self._r(np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
        r_down = self._r(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertLess(r_down, r_up)

    def test_swinging_at_target_energy_higher_than_static_down(self):
        """At down-rest with kinetic energy that brings the total to the upright energy
        target, the energy term saturates -> reward should beat the static down case
        (which has an energy deficit of 2 V_up)."""
        # Approximate kinetic energy needed.
        # Total energy at upright zero-velocity is +(m1+m2)g l1 + m2 g l2 = 1*9.81 + 0.5*9.81 = 14.715.
        # Down V is -14.715, so we need T ~= 29.43 to reach the target. Distribute
        # equally between theta-dots: T ~ 0.5 m l^2 omega^2 ~ omega^2 (with our params)
        # so omega ~ sqrt(29.43) ~ 5.4. Empirically we just need it bigger than down-rest.
        r_swinging = self._r(np.array([0.0, 0.0, 0.0, 0.0, 5.4, 5.4]))
        r_static_down = self._r(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertGreater(r_swinging, r_static_down)

    def test_kinetic_damping_at_upright(self):
        """When the angles are upright, motion *adds* kinetic energy and pushes
        :math:`E` away from the maximum potential -> reward strictly less than
        upright-rest."""
        r_still = self._r(np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
        r_moving = self._r(np.array([0.0, np.pi, np.pi, 0.0, 5.0, 5.0]))
        self.assertGreater(r_still, r_moving + 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
