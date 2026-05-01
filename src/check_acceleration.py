import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.env.double_pendulum import DoublePendulumCartEnv

def check_dynamics():
    env = DoublePendulumCartEnv(reset_mode="down")
    env.reset()
    
    print(f"Physics Params: M={env.M}, dt={env.dt}, Gain={env.velocity_gain}, MaxF={env.force_mag}")
    
    # Step Response: Target 1.0 m/s
    # Action = 1.0 / 10.0 = 0.1 (since max_vel=10)
    target_v = 1.0
    action = np.array([target_v / env.max_velocity])
    
    print(f"\n--- Step Response (Target v={target_v} m/s) ---")
    print("Step | Time | x | v | Force | a (approx)")
    
    prev_v = env.state[3]
    
    for i in range(10):
        state, _, _, _, _ = env.step(action)
        x = state[0]
        v = state[5] # x_dot
        
        # Calculate effective acceleration
        dv = v - prev_v
        a = dv / env.dt
        
        # Calculate Force (re-enact logic)
        # Note: This is what the env calculated internally
        err = target_v - prev_v
        force = env.velocity_gain * err
        force = np.clip(force, -env.force_mag, env.force_mag)
        
        print(f"{i+1:4d} | {env.dt*(i+1):.3f} | {x:.4f} | {v:.4f} | {force:.1f} | {a:.1f}")
        
        prev_v = v

if __name__ == "__main__":
    check_dynamics()
