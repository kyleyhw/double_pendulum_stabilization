import torch
import numpy as np
import sys
import os
import pygame
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

def debug_compare(log_dir="logs", duration=10.0):
    """
    Runs two identical simulations side-by-side to verify physics consistency.
    Left: "Overlay" Logic
    Right: "Final Run" Logic
    """
    print("--- Starting Debug Comparison ---")
    
    # 1. Find latest model
    import glob
    all_checkpoints = glob.glob(os.path.join(log_dir, "ppo_*_final.pth"))
    if not all_checkpoints:
        # Try finding any pth
        all_checkpoints = glob.glob(os.path.join(log_dir, "ppo_*_*.pth"))
        
    if not all_checkpoints:
        print("No checkpoints found.")
        return

    # Sort by time
    latest_ckpt = max(all_checkpoints, key=os.path.getctime)
    print(f"Using Model: {latest_ckpt}")
    
    # 2. Setup Environments
    # We want to prove that if we set the same seed, they are identical.
    seed = 42
    
    # Env A (Overlay Logic)
    env_a = DoublePendulumCartEnv(reset_mode="down", wind_std=0.0)
    env_a.set_curriculum(1.0) # Assume max difficulty for test
    
    # Env B (Final Run Logic)
    env_b = DoublePendulumCartEnv(reset_mode="down", wind_std=0.0)
    env_b.set_curriculum(1.0)
    
    # 3. Setup Agents
    state_dim = env_a.observation_space.shape[0]
    action_dim = env_a.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    agent.load(latest_ckpt)
    agent.policy.eval()
    
    # 4. Reset with SAME seed
    print(f"Resetting both environments with seed {seed}...")
    state_a, _ = env_a.reset(seed=seed, options={"mode": "down"})
    state_b, _ = env_b.reset(seed=seed, options={"mode": "down"})
    
    print(f"Initial State A: {state_a}")
    print(f"Initial State B: {state_b}")
    
    if not np.allclose(state_a, state_b):
        print("CRITICAL ERROR: Initial states differ despite same seed!")
        return
        
    # 5. Simulation Loop
    dt = env_a.dt
    steps = int(duration / dt)
    
    # Video Setup
    width, height = 800, 600
    # Output will be 1600x600 (Side by Side)
    out_width = width * 2
    out_height = height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "docs/images/debug_comparison.mp4"
    writer = cv2.VideoWriter(out_path, fourcc, 50.0, (out_width, out_height))
    
    # Visualizers
    viz_a = Visualizer(env_a, headless=True)
    viz_b = Visualizer(env_b, headless=True)
    
    stride = 4 # 200Hz -> 50Hz
    
    diffs = []
    
    print(f"Simulating {steps} steps...")
    for step in range(steps):
        # --- Step A (Overlay Logic) ---
        # In overlay, we do:
        # action, _ = agent.select_action(state, deterministic=True)
        # scaled_action = action * env.force_mag
        # next_state, ... = env.step(scaled_action)
        
        action_a_raw, _ = agent.select_action(state_a, deterministic=True)
        action_a_scaled = action_a_raw * env_a.force_mag
        next_state_a, reward_a, term_a, trunc_a, _ = env_a.step(action_a_scaled)
        
        # --- Step B (Final Run Logic) ---
        # In simulate.py (fixed), we do:
        # action, _ = agent.select_action(state, deterministic=True)
        # action = action * env.force_mag
        # next_state, ... = env.step(action)
        
        action_b_raw, _ = agent.select_action(state_b, deterministic=True)
        action_b_scaled = action_b_raw * env_b.force_mag
        next_state_b, reward_b, term_b, trunc_b, _ = env_b.step(action_b_scaled)
        
        # --- Compare ---
        diff = np.linalg.norm(next_state_a - next_state_b)
        diffs.append(diff)
        
        if diff > 1e-5:
            print(f"Divergence at Step {step}: {diff:.6f}")
            print(f"State A: {next_state_a}")
            print(f"State B: {next_state_b}")
            break
            
        state_a = next_state_a
        state_b = next_state_b
        
        if term_a or trunc_a: state_a, _ = env_a.reset(seed=seed)
        if term_b or trunc_b: state_b, _ = env_b.reset(seed=seed)
        
        # --- Render ---
        if step % stride == 0:
            viz_a.render(state_a, force=action_a_scaled[0], episode="Overlay", step=step, reward=reward_a, reward_fn_label="Overlay Logic")
            viz_b.render(state_b, force=action_b_scaled[0], episode="FinalRun", step=step, reward=reward_b, reward_fn_label="FinalRun Logic")
            
            frame_a = viz_a.get_frame() # RGB (H, W, 3)
            frame_b = viz_b.get_frame()
            
            # Convert to BGR for OpenCV
            frame_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2BGR)
            frame_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2BGR)
            
            # Concatenate
            combined = np.hstack((frame_a, frame_b))
            writer.write(combined)
            
    writer.release()
    pygame.quit()
    
    print(f"Comparison saved to {out_path}")
    max_diff = max(diffs) if diffs else 0.0
    print(f"Max State Difference: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("SUCCESS: Physics is identical.")
    else:
        print("FAILURE: Physics diverged.")

if __name__ == "__main__":
    debug_compare()
