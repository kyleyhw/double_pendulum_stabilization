import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv

class TestEnergyReward(unittest.TestCase):
    def setUp(self):
        self.env = DoublePendulumCartEnv()
        self.env.reset()
        
    def test_up_up_reward(self):
        """Test reward at the perfect Up-Up equilibrium."""
        # State: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        # Up-Up: theta1=pi, theta2=pi
        state = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
        self.env.state = state
        
        _, reward, _, _, _ = self.env.step(np.array([0.0]))
        
        print(f"Up-Up Reward: {reward}")
        # Should be close to 1.0 (Spatial=1, Energy=1)
        self.assertGreater(reward, 0.95)

    def test_down_down_static_reward(self):
        """Test reward at static Down-Down (low energy)."""
        # Down-Down: theta1=0, theta2=0
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.env.state = state
        
        _, reward, _, _, _ = self.env.step(np.array([0.0]))
        
        print(f"Down-Down Static Reward: {reward}")
        # Should be low (Spatial~0, Energy~0)
        self.assertLess(reward, 0.1)
        
    def test_down_down_high_energy_reward(self):
        """Test reward at Down-Down but with high kinetic energy (swinging)."""
        # We want Total Energy = Target Energy.
        # E_target = V_up
        # E_current = V_down + T
        # T = E_target - V_down
        
        # Calculate Target Energy
        m1, m2, l1, l2, g = self.env.m1, self.env.m2, self.env.l1, self.env.l2, self.env.g
        V_up = (m1 + m2) * g * l1 + m2 * g * l2
        V_down = -(m1 + m2) * g * l1 - m2 * g * l2
        
        Required_T = V_up - V_down
        
        # T = 0.5 * J * w^2 approx.
        # Let's just give it some velocity and check if reward is higher than static.
        # We don't need to be exact, just show that adding energy increases reward.
        
        # Give it some velocity
        state = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])
        self.env.state = state
        
        _, reward_dynamic, _, _, _ = self.env.step(np.array([0.0]))
        
        # Get static reward for comparison
        state_static = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.env.state = state_static
        _, reward_static, _, _, _ = self.env.step(np.array([0.0]))
        
        print(f"Down-Down Dynamic Reward: {reward_dynamic}")
        print(f"Down-Down Static Reward: {reward_static}")
        
        # Dynamic should be significantly higher due to Energy term
        self.assertGreater(reward_dynamic, reward_static + 0.1)

if __name__ == '__main__':
    unittest.main()
