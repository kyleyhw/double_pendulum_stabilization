import torch
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent

def analyze_model(log_dir="logs", num_episodes=5):
    # Find latest model
    checkpoints = glob.glob(os.path.join(log_dir, "ppo_*_*.pth"))
    if not checkpoints:
        print("No checkpoints found.")
        return

    # Sort by modification time to get the absolute latest file written
    latest_model = max(checkpoints, key=os.path.getmtime)
    print(f"Analyzing model: {latest_model}")

    env = DoublePendulumCartEnv(reset_mode="down")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    agent.load(latest_model)
    agent.policy.eval()
    
    max_heights = []
    episode_lengths = []
    all_actions = []
    
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        max_h = -999
        
        while not done:
            action, _ = agent.select_action(state)
            all_actions.append(action[0])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Calculate Tip Height (y coordinate of 2nd pole tip)
            # y is UP in world coords, but derivation had y DOWN?
            # Let's check env logic.
            # In visualizer: p1_y = cart_y + l1 * cos(theta1) (screen y is down)
            # Real world y: y = -l1 cos(theta1) - l2 cos(theta2) (if 0 is down)
            # Wait, in env:
            # t1 = (theta1 + pi) ...
            # Let's just use the angles directly.
            # If theta=0 is DOWN, then y_tip = -l1 cos(theta1) - l2 cos(theta2).
            # If theta=pi is UP, then y_tip = -l1 cos(pi) - ... = l1 + l2.
            # So Max Height is l1 + l2. Min is -(l1+l2).
            
            x, theta1, theta2, _, _, _ = next_state
            l1, l2 = env.l1, env.l2
            
            # Assuming theta=0 is DOWN (based on derivation context usually)
            # But let's check reset: reset_mode="down" -> theta=0.
            # So yes, 0 is down.
            # Height (y) relative to cart:
            y_tip = -l1 * np.cos(theta1) - l2 * np.cos(theta2)
            max_h = max(max_h, y_tip)
            
            state = next_state
            steps += 1
            if terminated or truncated:
                done = True
                
        max_heights.append(max_h)
        episode_lengths.append(steps)
        print(f"Episode {i+1}: Length={steps}, Max Height={max_h:.2f} m (Target: {l1+l2:.2f} m)")

    # Analysis
    print("\n--- Analysis ---")
    print(f"Avg Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Avg Max Height: {np.mean(max_heights):.2f} m")
    print(f"Action Mean: {np.mean(all_actions):.3f}")
    print(f"Action Std: {np.std(all_actions):.3f}")
    
    if np.mean(max_heights) > 1.5:
        print(">> The agent is swinging up high!")
    else:
        print(">> The agent is NOT swinging up high enough yet.")
        
    if np.std(all_actions) < 2.0:
        print(">> Exploration is low (Actions are clustered).")
    else:
        print(">> Exploration looks healthy (Actions are varied).")

if __name__ == "__main__":
    analyze_model()
