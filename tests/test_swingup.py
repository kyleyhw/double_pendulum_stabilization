r"""
Tests for the designed swing-up controller (:mod:`src.control.swingup`).

* The vectorised plant model reproduces the environment's RK4 step exactly.
* The committed nominal trajectory is dynamically consistent with the
  simulator (hanging at rest -> up-up at rest, within the cart bound).
* Discrete LQR holds the up-up equilibrium from an ``up`` reset.
* Closed loop: the full settle -> swing -> balance controller brings the
  chain from hanging to upright and holds it there at full difficulty.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.control.swingup import (  # noqa: E402
    PlantParams,
    SwingUpController,
    SwingUpTrajectory,
    optimize_swingup,
    rk4_step,
    state_error,
    wrap_angle,
)
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402

TRAJ_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "control", "data",
                         "swingup_d1.npz")


def _env(difficulty: float = 1.0) -> DoublePendulumCartEnv:
    env = DoublePendulumCartEnv(control_strategy=ForceControl())
    env.set_curriculum(difficulty)
    return env


def _pole_errors(env: DoublePendulumCartEnv) -> np.ndarray:
    return np.abs(wrap_angle(env.state[1:3] - np.pi))


class TestPlantModel(unittest.TestCase):
    def test_rk4_matches_env(self):
        rng = np.random.default_rng(0)
        for d in (0.3, 1.0):
            env = _env(d)
            p = PlantParams.from_env(env)
            for _ in range(20):
                s = rng.normal(size=6) * 2.0
                f = float(rng.normal() * 50.0)
                np.testing.assert_allclose(rk4_step(s, np.array(f), p),
                                           env._rk4_step(s.copy(), f, env.dt),
                                           rtol=1e-12, atol=1e-12)


class TestTrajectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traj = SwingUpTrajectory.load(TRAJ_PATH)
        cls.p = PlantParams.from_env(_env(1.0))

    def test_physics_match_full_difficulty(self):
        self.assertEqual(self.traj.params, self.p)

    def test_boundary_conditions(self):
        np.testing.assert_allclose(self.traj.states[0], np.zeros(6), atol=1e-8)
        np.testing.assert_allclose(state_error(self.traj.states[-1], self.traj.goal),
                                   np.zeros(6), atol=1e-6)
        self.assertTrue(np.all(np.abs(self.traj.goal[1:3]) == np.pi))
        # x_max = 2.5 m on the knots; the feasibility polish may nudge it by < 5 cm.
        self.assertLess(np.abs(self.traj.states[:, 0]).max(), 2.55)

    def test_dynamically_consistent(self):
        nxt = rk4_step(self.traj.states[:-1], self.traj.forces, self.p)
        np.testing.assert_allclose(nxt, self.traj.states[1:], atol=1e-6)

    def test_optimizer_short_horizon_smoke(self):
        # A tiny problem (hanging -> hanging displaced by 0.2 m) must solve to
        # a dynamically consistent trajectory.
        goal = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        traj = optimize_swingup(self.p, duration=1.0, n_sub=10, goal=goal, maxiter=100)
        self.assertLess(traj.info["max_defect"], 1e-8)
        np.testing.assert_allclose(traj.states[-1], goal, atol=1e-6)


class TestClosedLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traj = SwingUpTrajectory.load(TRAJ_PATH)

    def _run(self, env, ctrl, seconds):
        for _ in range(int(round(seconds / env.dt))):
            _, _, terminated, _, _ = env.step(ctrl.action(env.state))
            self.assertFalse(terminated)

    def test_balance_from_up(self):
        env = _env(1.0)
        ctrl = SwingUpController(self.traj)
        env.reset(seed=3, options={"mode": "up"})
        ctrl.reset(env.state)
        self.assertEqual(ctrl.phase, "balance")
        self._run(env, ctrl, 3.0)
        self.assertTrue(np.all(_pole_errors(env) < 0.02))

    def test_swing_up_from_hanging(self):
        env = _env(1.0)
        ctrl = SwingUpController(self.traj)
        for seed in (0, 1):
            env.reset(seed=seed, options={"mode": "down"})
            ctrl.reset(env.state)
            self._run(env, ctrl, 3.0 + self.traj.duration + 2.0)
            self.assertEqual(ctrl.phase, "balance")
            self.assertTrue(np.all(_pole_errors(env) < 0.05), _pole_errors(env))
            self.assertLess(abs(env.state[0]), env.x_soft)


if __name__ == "__main__":
    unittest.main()
