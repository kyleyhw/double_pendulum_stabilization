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
        # Should be close to 1.0 (Spatial=1, Energy=1, Kinetic=1)
        # 0.4*1 + 0.4*1 + 0.2*1 = 1.0
        self.assertGreater(reward, 0.95)

    def test_down_down_static_reward(self):
        """Test reward at static Down-Down (low energy)."""
        # theta1=0, theta2=0, velocities=0
        # Energy is minimal (not target). Spatial is far.
        # Kinetic is perfect (0 velocity), BUT Gating should kill it.
        # Gating = exp(-(pi^2 + pi^2)) approx exp(-20) approx 0.
        # So Reward should be very close to 0.
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.env.state = state
        _, reward, _, _, _ = self.env.step(np.array([0.0]))
        
        print(f"Down-Down Static Reward: {reward}")
        # Should be low (wrong position, wrong energy, gated kinetic)
        self.assertLess(reward, 0.05)

    def test_down_down_high_energy_reward(self):
        """Test reward at Down-Down but with high kinetic energy (swinging)."""
        # This simulates the bottom of a swing that has enough energy to reach top.
        # Energy ~ Target. Spatial ~ 0. Kinetic ~ 0 (high velocity).
        # Reward ~ w_e * 1.0 = 0.4.
        
        # We need to find velocities that give E_total = E_target at bottom.
        # V_bottom = -(m1+m2)gl1 - m2gl2. V_top = (m1+m2)gl1 + m2gl2.
        # Delta V = 2 * V_top.
        # So T must be 2 * V_top.
        
        # Let's just manually set a state with correct energy.
        # We can use the environment to find it or just approximate.
        # For the test, we just want to see R_dynamic > R_static.
        
        # State from previous run that had high reward:
        # We need T approx 29.4J.
        # T = 1.25 * w^2. w^2 = 23.5. w ~ 4.85.
        state = np.array([0.0, 0.0, 0.0, 0.0, 4.85, 4.85]) 
        self.env.state = state
        _, reward_dynamic, _, _, _ = self.env.step(np.array([0.0]))
        
        # Get static reward again for comparison
        self.env.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, reward_static, _, _, _ = self.env.step(np.array([0.0]))
        
        print(f"Down-Down Dynamic Reward: {reward_dynamic}")
        print(f"Down-Down Static Reward: {reward_static}")
        
        # Dynamic (Swinging with Energy) should be higher than Static (Sitting)
        # 0.4 (Energy) > 0.2 (Kinetic)
        self.assertGreater(reward_dynamic, reward_static + 0.1)

    def test_kinetic_damping(self):
        """Test that high kinetic energy is penalized even if total energy is correct."""
        # Case A: Upright and Still (Ideal)
        # theta1=pi, theta2=pi, velocities=0
        state_a = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
        self.env.state = state_a
        _, r_a, _, _, _ = self.env.step(np.array([0.0]))
        
        # Case B: Upright but Moving Fast (Flailing)
        # theta1=pi, theta2=pi, velocities=high
        # Note: This increases Total Energy, so E_reward will drop too.
        # To isolate Kinetic Reward, we need a state with SAME Total Energy but different T/V split?
        # No, the point of Kinetic Damping is to force T->0.
        # So we just compare Upright+Still vs Upright+Moving.
        state_b = np.array([0.0, np.pi, np.pi, 0.0, 5.0, 5.0])
        self.env.state = state_b
        _, r_b, _, _, _ = self.env.step(np.array([0.0]))
        
        print(f"\nKinetic Damping Test:")
        print(f"  Reward (Still): {r_a:.4f}")
        print(f"  Reward (Moving): {r_b:.4f}")
        
        # Expect R_still > R_moving significantly
        self.assertGreater(r_a, r_b + 0.2, "Static upright state should have significantly higher reward than moving upright state.")

    def test_kinetic_damping_basin(self):
        """Test that the reward basin is wide enough to provide gradient at high velocities."""
        # Case: Upright but Moving Very Fast (T=10.0)
        # Old Reward: exp(-10) ~ 4.5e-5 (Zero gradient)
        # New Reward: 0.2*exp(-1) + 0.8*exp(-10) ~ 0.07 (Non-zero)
        
        # We need T=10.0.
        # T = 0.5 * m * v^2 approx. Let's just force T calculation or find state.
        # Let's use a state with high velocity.
        # v=4.5 -> T approx 25.
        
        state = np.array([0.0, np.pi, np.pi, 0.0, 4.5, 4.5])
        self.env.state = state
        _, reward, _, _, _ = self.env.step(np.array([0.0]))
        
        print(f"High Velocity Reward (Basin Check): {reward}")
        # Should be non-trivial (> 0.01) to provide gradient
        self.assertGreater(reward, 0.01, "Reward should be non-zero even at high velocities to provide a gradient.")

if __name__ == '__main__':
    unittest.main()
