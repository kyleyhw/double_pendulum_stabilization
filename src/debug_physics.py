import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.env.double_pendulum import DoublePendulumCartEnv

def test_force(force_val):
    print(f"\n--- Testing Force: {force_val} N ---")
    env = DoublePendulumCartEnv(reset_mode="down")
    env.reset()
    
    # Force the force_mag to be high enough to allow this force
    env.force_mag = max(abs(force_val), 5000.0)
    
    # Step 1
    # env.step expects action to be in Newtons if we pass it directly?
    # NO. env.step clips action[0] to [-force_mag, force_mag].
    # So we should pass the Force Value directly.
    action = np.array([force_val]) 
    
    print(f"Initial State: x={env.state[0]:.4f}, v={env.state[3]:.4f}")
    
    for i in range(5):
        state, reward, term, trunc, _ = env.step(action)
        x = state[0]
        v = state[5] # x_dot is index 5 in observation?
        # Observation: [x, sin, cos, sin, cos, x_dot, t1_dot, t2_dot]
        # Index 0: x
        # Index 5: x_dot
        print(f"Step {i+1}: x={x:.4f}, v={v:.4f}")

if __name__ == "__main__":
    test_force(10.0)
    test_force(100.0)
    test_force(5000.0)
