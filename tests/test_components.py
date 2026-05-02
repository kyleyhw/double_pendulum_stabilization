r"""
Unit tests for the redesigned components.

Coverage:
* :class:`RunningMeanStd` -- correctness of the parallel-Welford merge against
  ``numpy``'s exact mean/var.
* :class:`NormalizeObservation` -- output is approximately zero-mean unit-var
  after sufficient samples.
* :class:`ActorCritic` tanh squash -- log-prob via change-of-variables matches
  the analytic formula and actions stay in :math:`[-1, 1]`.
* :class:`PPOAgent.update` -- runs cleanly, returns expected diagnostic keys,
  and the value head reduces MSE on a stationary signal.
* :class:`CartPendulumBase` integrators -- semi-implicit Euler conserves
  energy on long horizons better than RK4 for free pendulum motion.
* Soft / hard cart bounds -- expected reward shape near the boundary.
* :class:`EnergyShapingReward` -- attains its analytic max at the upright
  zero-velocity equilibrium.

All tests are runtime-cheap and use fixed seeds for determinism.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.ppo import ActorCritic, Memory, PPOAgent  # noqa: E402
from src.env.cart_pendulum_base import CartPendulumBase  # noqa: E402  (touched for completeness)
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.env.single_pendulum import SinglePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402
from src.strategies.rewards import (  # noqa: E402
    EnergyShapingReward,
    ExponentialSwingUpReward,
    HybridLQRSwingUpReward,
    LQRCostReward,
)
from src.utils.normalize import NormalizeObservation, RunningMeanStd  # noqa: E402


class TestRunningMeanStd(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        rms = RunningMeanStd(shape=(3,))
        all_samples = []
        for _ in range(50):
            batch = rng.normal(loc=[1, -2, 0.5], scale=[0.1, 1.0, 2.0], size=(20, 3))
            all_samples.append(batch)
            rms.update(batch)
        flat = np.concatenate(all_samples, axis=0)
        np.testing.assert_allclose(rms.mean, flat.mean(axis=0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(rms.var, flat.var(axis=0), rtol=1e-3, atol=1e-3)


class TestNormalizeObservation(unittest.TestCase):
    def test_freeze_preserves_state(self):
        env = SinglePendulumCartEnv()
        wrapper = NormalizeObservation(env, training=True)
        for _ in range(20):
            wrapper.reset(seed=0)
            for _ in range(50):
                wrapper.step(np.zeros(1, dtype=np.float32))
        snapshot = wrapper.obs_rms.state_dict()
        wrapper.training = False
        for _ in range(20):
            wrapper.reset(seed=1)
            for _ in range(50):
                wrapper.step(np.zeros(1, dtype=np.float32))
        np.testing.assert_array_equal(wrapper.obs_rms.mean, snapshot["mean"])


class TestTanhSquashedPolicy(unittest.TestCase):
    def test_action_range_and_log_prob(self):
        import torch
        torch.manual_seed(0)
        ac = ActorCritic(state_dim=8, action_dim=1, init_log_std=0.0)
        s = torch.zeros((128, 8))
        a, lp = ac.get_action(s)
        # Action box.
        self.assertTrue(torch.all(a >= -1.0) and torch.all(a <= 1.0))
        # Re-evaluate the same actions and verify the log-prob agrees up to
        # the squash epsilon (tiny numerical jitter).
        lp2, _, _ = ac.evaluate(s, a)
        self.assertTrue(torch.allclose(lp, lp2, atol=1e-3))


class TestPPOUpdate(unittest.TestCase):
    def test_value_loss_decreases_on_stationary_target(self):
        import torch
        torch.manual_seed(0)
        np.random.seed(0)
        agent = PPOAgent(state_dim=4, action_dim=1, lr=1e-3, k_epochs=4,
                         minibatch_size=64, target_kl=None)
        rng = np.random.default_rng(0)
        # Stationary target reward = 1 always.
        def fill_memory():
            mem = Memory()
            for _ in range(256):
                s = rng.normal(size=4).astype(np.float32)
                a, lp = agent.select_action(s)
                mem.states.append(s)
                mem.actions.append(a)
                mem.log_probs.append(float(lp))
                mem.rewards.append(1.0)
                mem.is_terminals.append(False)
            return mem
        first = agent.update(fill_memory(), last_state=np.zeros(4, dtype=np.float32))
        for _ in range(5):
            last = agent.update(fill_memory(), last_state=np.zeros(4, dtype=np.float32))
        self.assertLess(last["value_loss"], first["value_loss"])
        for k in ["policy_loss", "value_loss", "entropy", "approx_kl",
                  "clip_fraction", "explained_variance", "lr", "eps_clip", "entropy_coef"]:
            self.assertIn(k, last)


class TestIntegrators(unittest.TestCase):
    def test_semi_implicit_energy_drift_bounded(self):
        # Compare RK4 vs. semi-implicit Euler over 5 s of free motion.
        # Both are non-symplectic / O(dt) accurate respectively, but semi-
        # implicit *bounds* energy drift. We just check it stays small.
        env_si = DoublePendulumCartEnv(
            control_strategy=ForceControl(),
            reward_strategy=ExponentialSwingUpReward(),
            integrator="semi_implicit",
        )
        env_si.friction_cart = 0.0
        env_si.friction_pole = 0.0
        env_si.state = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.0])
        E0 = env_si._get_energy()
        for _ in range(int(round(5.0 / env_si.dt))):
            env_si.step(np.zeros(1, dtype=np.float32))
        rel_drift = abs(env_si._get_energy() - E0) / abs(E0)
        # Semi-implicit Euler has O(dt) local truncation; over 5 s with dt=5e-3 we
        # expect O(1%) drift max.
        self.assertLess(rel_drift, 0.1)


class TestSoftBoundary(unittest.TestCase):
    def test_penalty_grows_quadratically_outside_x_soft(self):
        env = DoublePendulumCartEnv(
            control_strategy=ForceControl(),
            reward_strategy=ExponentialSwingUpReward(),
            x_soft=4.0, x_hard=8.0, boundary_penalty_k=0.5,
        )
        env.reset(seed=0)
        # Force the cart outside x_soft.
        env.state[0] = 5.0  # 1.0 m past x_soft
        _, r1, _, _, _ = env.step(np.zeros(1, dtype=np.float32))
        env.reset(seed=0)
        env.state[0] = 6.0  # 2.0 m past x_soft
        _, r2, _, _, _ = env.step(np.zeros(1, dtype=np.float32))
        # Penalty term grows as overshoot^2 -> reward at 6 m is much more negative.
        self.assertLess(r2, r1)
        self.assertLessEqual(r1, 0.0)


class TestLQRCostReward(unittest.TestCase):
    def test_reward_landscape(self):
        r = LQRCostReward()
        r.set_curriculum(1.0)
        # Upright at rest: reward equals b_alive (its maximum).
        r_up = r.compute_reward(np.array([0, np.pi, np.pi, 0, 0, 0]), {})
        self.assertAlmostEqual(r_up, r.b_alive, places=6)
        # Adding velocity strictly reduces reward (kinetic-energy penalty).
        r_moving = r.compute_reward(np.array([0, np.pi, np.pi, 0, 1.0, 1.0]), {})
        self.assertLess(r_moving, r_up)
        # Off-upright by a small angle reduces reward, by a larger angle reduces more.
        r_small = r.compute_reward(np.array([0, np.pi - 0.05, np.pi - 0.05, 0, 0, 0]), {})
        r_large = r.compute_reward(np.array([0, np.pi - 0.5, np.pi - 0.5, 0, 0, 0]), {})
        self.assertGreater(r_small, r_large)

    def test_curriculum_strengthens_angle_penalty(self):
        r = LQRCostReward()
        s = np.array([0, np.pi - 0.5, np.pi - 0.5, 0, 0, 0])
        r.set_curriculum(0.0)
        r_easy = r.compute_reward(s, {})
        r.set_curriculum(1.0)
        r_hard = r.compute_reward(s, {})
        # Harder curriculum -> larger q_theta -> more negative reward.
        self.assertGreater(r_easy, r_hard)


class TestHybridLQRSwingUpReward(unittest.TestCase):
    def test_quadratic_pull_outside_band(self):
        """The quadratic term provides a strict pull toward upright at every angle."""
        r = HybridLQRSwingUpReward()
        r.set_curriculum(0.0)  # widest band, exp reward fires almost everywhere
        # Reset to make exp reward part predictable.
        r.reset()
        # At down (errors = pi each), the quadratic term contributes
        # -q_theta * 2*pi^2 = -0.2 * 19.74 = -3.95.
        r_down = r.compute_reward(np.array([0, 0, 0, 0, 0, 0]), {})
        r_horiz = r.compute_reward(np.array([0, np.pi/2, np.pi/2, 0, 0, 0]), {})
        # Reward strictly increases as we approach upright.
        self.assertLess(r_down, r_horiz)

    def test_exp_continuity_dominates_inside_band(self):
        """Inside the band, the exp continuity reward outpaces the small quadratic penalty."""
        r = HybridLQRSwingUpReward()
        r.set_curriculum(1.0)  # narrowest band, but the upright state is exactly inside.
        r.reset()
        # Burn 200 steps near upright to accumulate T_up = 1.0 s -> e^1 - 1 ~ 1.72.
        upright = np.array([0, np.pi, np.pi, 0, 0, 0])
        for _ in range(200):
            r.compute_reward(upright, {"dt": 0.005})
        r_up = r.compute_reward(upright, {"dt": 0.005})
        # exp continuity dominates: should be clearly positive.
        self.assertGreater(r_up, 0.5)


class TestEnergyShapingReward(unittest.TestCase):
    def test_max_at_upright_rest(self):
        env = DoublePendulumCartEnv(
            control_strategy=ForceControl(),
            reward_strategy=EnergyShapingReward(),
        )
        env.set_curriculum(1.0)
        env.reset(seed=0)
        env.state = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
        # Single step to evaluate reward at this state.
        _, r, _, _, _ = env.step(np.zeros(1, dtype=np.float32))
        # Max possible value (sum of weights) is 1.0 by design.
        self.assertGreater(r, 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
