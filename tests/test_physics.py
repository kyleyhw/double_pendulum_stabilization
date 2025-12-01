import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env.double_pendulum import DoublePendulumCartEnv

class TestDoublePendulumPhysics(unittest.TestCase):
    def setUp(self):
        self.env = DoublePendulumCartEnv()
        
    def calculate_energy(self, state):
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
        
        M, m1, m2 = self.env.M, self.env.m1, self.env.m2
        l1, l2 = self.env.l1, self.env.l2
        g = self.env.g
        
        # Kinetic Energy
        # v_c^2 = x_dot^2
        # v_1^2 = ...
        # v_2^2 = ...
        
        # From derivation:
        # T = 0.5*M*x_dot^2 + 0.5*m1*v1^2 + 0.5*m2*v2^2
        
        v1_sq = x_dot**2 + 2*x_dot*l1*theta1_dot*np.cos(theta1) + l1**2*theta1_dot**2
        v2_sq = (x_dot**2 + l1**2*theta1_dot**2 + l2**2*theta2_dot**2 + 
                 2*x_dot*l1*theta1_dot*np.cos(theta1) + 
                 2*x_dot*l2*theta2_dot*np.cos(theta2) + 
                 2*l1*l2*theta1_dot*theta2_dot*np.cos(theta1 - theta2))
                 
        T = 0.5 * M * x_dot**2 + 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        
        # Potential Energy
        # V = -(m1+m2)g l1 cos(theta1) - m2 g l2 cos(theta2)
        V = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
        
        return T + V

    def test_energy_conservation(self):
        """Test that total energy is conserved with zero external force."""
        # Initialize in a general state
        # x=0, theta1=1.0, theta2=2.0, velocities=0
        state = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.0])
        self.env.state = state
        
        initial_energy = self.calculate_energy(state)
        
        # Run simulation for 1 second (50 steps at dt=0.02)
        energies = []
        for _ in range(50):
            # Apply zero force
            obs, _, _, _, _ = self.env.step(np.array([0.0]))
            energy = self.calculate_energy(self.env.state)
            energies.append(energy)
            
        # Check conservation
        # RK4 is not symplectic, so energy will drift slightly, but should be small.
        # Allow 0.1% error over 1 second.
        max_deviation = np.max(np.abs(np.array(energies) - initial_energy))
        relative_error = max_deviation / np.abs(initial_energy)
        
        print(f"\nInitial Energy: {initial_energy:.4f}")
        print(f"Max Deviation: {max_deviation:.6f}")
        print(f"Relative Error: {relative_error:.6%}")
        
        self.assertLess(relative_error, 1e-3, "Energy drift exceeded 0.1%")

if __name__ == '__main__':
    unittest.main()
