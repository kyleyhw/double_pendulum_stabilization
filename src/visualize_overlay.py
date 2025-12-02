import torch
import numpy as np
import sys
import os
import glob
import pygame
import cv2
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

    # 1. Find Checkpoints from the LATEST run
    all_checkpoints = glob.glob(os.path.join(log_dir, "ppo_*_*.pth"))
    if not all_checkpoints:
        print("No checkpoints found.")
        return

    # Extract run IDs (YYYYMMDD_HHMMSS)
    run_ids = set()
    for f in all_checkpoints:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 3:
            run_id = f"{parts[1]}_{parts[2]}"
            run_ids.add(run_id)
    
    if not run_ids:
        print("No valid run IDs found.")
        return

    # Sort run IDs to find latest
    latest_run_id = sorted(list(run_ids))[-1]
    print(f"Using checkpoints from latest run: {latest_run_id}")
    
    checkpoints = [f for f in all_checkpoints if latest_run_id in f]
    
    # Load Log File to get Difficulty
    log_file = os.path.join(log_dir, f"training_log_{latest_run_id}.csv")
    episode_difficulty = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                header = f.readline().strip().split(',')
                diff_idx = header.index("Difficulty")
                ep_idx = header.index("Episode")
                
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) > diff_idx:
                        ep = int(parts[ep_idx])
                        diff = float(parts[diff_idx])
                        episode_difficulty[ep] = diff
            print(f"Loaded difficulty map for {len(episode_difficulty)} episodes.")
        except Exception as e:
            print(f"Failed to load log file: {e}")
    else:
        print(f"Log file not found: {log_file}")
    
    def get_episode(f):
        try:
            val = f.split('_')[-1].split('.')[0]
            if val == "final": return 999999999
            return int(val)
        except:
            return 0

    checkpoints.sort(key=get_episode)
    
    if len(checkpoints) == 0:
        print("No checkpoints found.")
        return

    # Select N evenly spaced checkpoints
    # Always include the first and last
    if len(checkpoints) > num_checkpoints:
        indices = np.linspace(0, len(checkpoints)-1, num_checkpoints, dtype=int)
        selected_checkpoints = [checkpoints[i] for i in indices]
    else:
        selected_checkpoints = checkpoints

    print(f"Selected checkpoints: {[os.path.basename(c) for c in selected_checkpoints]}")

    # 2. Initialize Environments and Agents
    envs = []
    agents = []
    states = []
    
    # Use headless mode for Visualizer to avoid window issues during generation
    base_env = DoublePendulumCartEnv(reset_mode="down")
    viz = Visualizer(base_env, headless=True)
    
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    for cp in selected_checkpoints:
        # Determine Difficulty
        ep_num = get_episode(os.path.basename(cp))
        # If final, find max episode in log? Or just use 1.0?
        # Let's try to find exact match, or closest lower match.
        
        difficulty = 0.0
        if ep_num in episode_difficulty:
            difficulty = episode_difficulty[ep_num]
        elif ep_num == 999999999: # Final
             if episode_difficulty:
                 difficulty = episode_difficulty[max(episode_difficulty.keys())]
        else:
            # Find closest
            sorted_eps = sorted(episode_difficulty.keys())
            for e in sorted_eps:
                if e <= ep_num:
                    difficulty = episode_difficulty[e]
                else:
                    break
        
        print(f"Checkpoint {os.path.basename(cp)} (Ep {ep_num}) -> Difficulty {difficulty:.2f}")
        
        env = DoublePendulumCartEnv(reset_mode="down")
        env.set_curriculum(difficulty) # Set Correct Physics!
        
        state, _ = env.reset(seed=42) # Same seed for all to compare behavior!
        envs.append(env)
        states.append(state)
        
        agent = PPOAgent(state_dim, action_dim)
        agent.load(cp)
        agent.policy.eval()
        agents.append(agent)

    # Reward Histories
    reward_histories = [[] for _ in range(len(agents))]

    # Colors: Fade from Red (early) to Green (late)
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('viridis') # Blue to Green/Yellow
    colors = [cmap(i) for i in np.linspace(0, 1, len(selected_checkpoints))]
    # Convert to 0-255 RGB
            print(f"Montage saved to {output_mp4}")
        viz.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_mp4", action="store_true", help="Save MP4")
    parser.add_argument("--output", type=str, default="docs/images/overlay_montage.mp4")
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    
    visualize_overlay(save_mp4=args.save_mp4, output_mp4=args.output, duration=args.duration)
