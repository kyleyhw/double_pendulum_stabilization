r"""
Tests for equilibrium switching (:mod:`src.control.switching`) and the Phase S /
Phase 6 RL trainer's building blocks (:mod:`src.train_balance`).

* Every stored transition starts at its source equilibrium, ends at its target
  (angles mod :math:`2\pi`), and is dynamically consistent with the simulator.
* Closed loop: the switching controller performs a stable -> unstable and an
  unstable -> unstable transition on the real environment.
* Trainer pieces: observation layout / goal one-hot, reward maximised at the
  goal, and the reverse-curriculum sampler's reach semantics.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.control.swingup import SwingUpTrajectory, rk4_step, state_error, wrap_angle  # noqa: E402
from src.control.switching import (  # noqa: E402
    NAMES,
    SwitchingController,
    equilibrium,
    goal_candidates,
    load_library,
)
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402
from src.train_balance import (  # noqa: E402
    DEFAULT_TRAJ,
    GOAL_ANGLES,
    InitSampler,
    balance_reward,
    make_obs,
)


class TestSwitchingLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_library()

    def test_all_pairs_present(self):
        self.assertEqual(len(self.lib), 12)

    def test_boundary_conditions_and_consistency(self):
        for (a, b), t in self.lib.items():
            np.testing.assert_allclose(state_error(t.states[0], equilibrium(a)), 0, atol=1e-6)
            np.testing.assert_allclose(state_error(t.states[-1], equilibrium(b)), 0, atol=1e-5)
            nxt = rk4_step(t.states[:-1], t.forces, t.params)
            np.testing.assert_allclose(nxt, t.states[1:], atol=1e-6, err_msg=f"{a}->{b}")

    def test_goal_candidates(self):
        self.assertEqual(len(goal_candidates("DD", "UU")), 4)
        self.assertEqual(len(goal_candidates("DD", "DU")), 2)
        for g in goal_candidates("UU", "UD"):
            self.assertEqual(g[1], np.pi)  # pole 1 stays up


class TestSwitchingClosedLoop(unittest.TestCase):
    def _transition(self, src, dst, seed=0):
        lib = load_library()
        ctrl = SwitchingController(lib, initial=src)
        env = DoublePendulumCartEnv(control_strategy=ForceControl())
        env.set_curriculum(1.0)
        env.reset(seed=seed)
        env.state = equilibrium(src) + np.random.default_rng(seed).uniform(-0.02, 0.02, 6)
        steps = int(round((1.0 + lib[(src, dst)].duration + 2.0) / env.dt))
        for j in range(steps):
            if j == int(round(1.0 / env.dt)):
                ctrl.request(dst)
            env.step(ctrl.action(env.state))
            self.assertLess(abs(env.state[0]), env.x_soft)
        self.assertTrue(ctrl.settled)
        err = np.abs(wrap_angle(env.state[1:3] - equilibrium(dst)[1:3]))
        self.assertTrue(np.all(err < 0.05), err)

    def test_dd_to_du(self):
        self._transition("DD", "DU")

    def test_uu_to_ud(self):
        self._transition("UU", "UD")


class TestBalanceTrainerPieces(unittest.TestCase):
    def test_obs_layout_matches_env(self):
        env = DoublePendulumCartEnv(control_strategy=ForceControl())
        obs, _ = env.reset(seed=0)
        ours = make_obs(env.state)[0]
        scale = np.array([0.5, 1, 1, 1, 1, 1 / 3, 1 / 8, 1 / 8])
        np.testing.assert_allclose(ours, obs * scale, rtol=1e-6, atol=1e-6)

    def test_goal_onehot(self):
        o = make_obs(np.zeros((3, 6)), np.array([0, 2, 3]))
        self.assertEqual(o.shape, (3, 12))
        np.testing.assert_array_equal(o[:, 8:], np.eye(4)[[0, 2, 3]])

    def test_reward_peaks_at_goal(self):
        for g, name in enumerate(NAMES):
            at_goal = np.zeros((1, 6))
            at_goal[0, 1:3] = GOAL_ANGLES[g]
            off = at_goal.copy()
            off[0, 1] += 0.3
            r_at = balance_reward(at_goal, 3.5, np.array([g]))[0]
            self.assertAlmostEqual(r_at, 1.0, places=9, msg=name)
            self.assertLess(balance_reward(off, 3.5, np.array([g]))[0], r_at)

    def test_sampler_reach(self):
        traj = SwingUpTrajectory.load(DEFAULT_TRAJ)
        smp = InitSampler({1: [traj]}, np.random.default_rng(0), tau_step=0.5)
        s, goals = smp.sample(64)
        self.assertTrue(np.all(goals == 1))
        self.assertTrue(np.all(np.abs(wrap_angle(s[:, 1:3] - np.pi)) <= 0.05 + 1e-12))
        while smp.reach <= 0.0:
            smp.advance()
        self.assertEqual(smp.r, smp.r_max)
        for _ in range(20):
            smp.advance()
        self.assertTrue(smp.done)
        s, _, reach = smp.frontier(8)
        self.assertAlmostEqual(reach, traj.duration)
        np.testing.assert_allclose(s, np.broadcast_to(traj.states[0], s.shape), atol=0.05 + 1e-9)
        smp.load_state_dict({"r": 0.25, "tau_min": 3.0})  # legacy single-goal checkpoint
        self.assertAlmostEqual(smp.reach, traj.duration - 3.0)


if __name__ == "__main__":
    unittest.main()
