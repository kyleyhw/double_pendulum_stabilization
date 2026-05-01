import unittest
import numpy as np
import sys
import os

# Add project root to path so `src.*` imports resolve.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.strategies.controls import ForceControl
from src.strategies.rewards import DoublePendulumStandardReward

class TestDoublePendulumPhysics(unittest.TestCase):
    def setUp(self):
        # Use ForceControl with action=0 -> F=0 so we measure pure physics.
        # VelocityControl applies a P-controller to cart velocity, which would
        # inject force whenever the cart drifts and so spoil the energy-conservation test.
        self.env = DoublePendulumCartEnv(
            control_strategy=ForceControl(),
            reward_strategy=DoublePendulumStandardReward(),
        )
        # Reset friction to zero (pure physics; default already 0 but make explicit).
        self.env.friction_cart = 0.0
        self.env.friction_pole = 0.0
        
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
        """Total mechanical energy is conserved under zero external force.

        Because RK4 is not symplectic, we expect bounded but non-zero drift over
        a finite horizon. We integrate for one physical second (= 1 / dt steps,
        i.e. 200 steps at dt = 0.005 s) and require <0.1% relative drift.
        """
        state = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.0])
        self.env.state = state.copy()
        self.env.reward_strategy.reset()

        initial_energy = self.calculate_energy(state)

        # 1 second of integration at the env's actual dt.
        n_steps = int(round(1.0 / self.env.dt))
        energies = []
        for _ in range(n_steps):
            self.env.step(np.array([0.0]))  # ForceControl: action=0 -> F=0.
            energies.append(self.calculate_energy(self.env.state))

        max_deviation = float(np.max(np.abs(np.array(energies) - initial_energy)))
        relative_error = max_deviation / float(np.abs(initial_energy))

        print(f"\nInitial Energy: {initial_energy:.4f}")
        print(f"Max Deviation:  {max_deviation:.6f}")
        print(f"Relative Error: {relative_error:.6%}")

        self.assertLess(relative_error, 1e-3, "Energy drift exceeded 0.1%")

if __name__ == '__main__':
    unittest.main()
