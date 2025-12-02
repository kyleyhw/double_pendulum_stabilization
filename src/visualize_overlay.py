import torch
import numpy as np
import sys
import os
import glob
import pygame
import cv2
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

def visualize_overlay(log_dir="logs", num_checkpoints=5, duration=20.0, save_mp4=False, output_mp4="overlay.mp4", seed=42):
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
        
    print(f"Found Run IDs: {sorted(list(run_ids))}")
    latest_run_id = sorted(list(run_ids))[-1]
    print(f"Visualizing Run ID: {latest_run_id}")
    
    # Filter checkpoints for this run
    run_checkpoints = [f for f in all_checkpoints if latest_run_id in f]
    
    # Sort by episode number
    def get_ep(f):
        try:
            val = os.path.basename(f).split('_')[-1].split('.')[0]
            if val == "final": return 999999999
            return int(val)
        except:
            return 0
            
    run_checkpoints.sort(key=get_ep)
    
    # Select N evenly spaced checkpoints (Unique only)
    # Always include the LAST checkpoint (which might be 'final')
    if not run_checkpoints:
        selected_checkpoints = []
    else:
        # If we have fewer checkpoints than requested, take all
        if len(run_checkpoints) <= num_checkpoints:
            selected_checkpoints = run_checkpoints
        else:
            # We want N checkpoints. One MUST be the last one.
            # So we select N-1 from the rest.
            indices = np.linspace(0, len(run_checkpoints)-2, num_checkpoints-1, dtype=int)
            indices = np.unique(indices)
            selected_checkpoints = [run_checkpoints[i] for i in indices]
            selected_checkpoints.append(run_checkpoints[-1]) # Add the last one
            
    # Ensure uniqueness just in case
    seen = set()
    unique_checkpoints = []
    for ckpt in selected_checkpoints:
        if ckpt not in seen:
            unique_checkpoints.append(ckpt)
            seen.add(ckpt)
    selected_checkpoints = unique_checkpoints
        
    print(f"Selected Checkpoints: {[os.path.basename(f) for f in selected_checkpoints]}")
    
    # Load Training Log to get Difficulty
    log_file = os.path.join(log_dir, f"training_log_{latest_run_id}.csv")
    episode_difficulty_map = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                header = f.readline().strip().split(',')
                try:
                    diff_idx = header.index("Difficulty")
                    ep_idx = header.index("Episode")
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) > diff_idx:
                            episode_difficulty_map[int(parts[ep_idx])] = float(parts[diff_idx])
                except ValueError:
                    print("Could not find Difficulty/Episode columns in log.")
        except Exception as e:
            print(f"Error reading log file: {e}")

    # Initialize Environment and Agents
    # Note: We initialize with default physics, but will update per-agent in the loop
    env = DoublePendulumCartEnv(reset_mode="down", wind_std=0.0)
    # Headless mode for server rendering
    visualizer = Visualizer(env, headless=True) 
    
    agents = []
    envs = []
    difficulties = []
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    for ckpt in selected_checkpoints:
        # Create Env for this agent
        agent_env = DoublePendulumCartEnv(reset_mode="down", wind_std=0.0)
        envs.append(agent_env)
        
        agent = PPOAgent(state_dim, action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2, k_epochs=4)
        agent.load(ckpt)
        agent.policy.eval()
        agents.append(agent)
        
        # Determine difficulty
        ep_num = get_ep(ckpt)
        # If final, use max difficulty found
        if ep_num == 999999999 and episode_difficulty_map:
             diff = episode_difficulty_map[max(episode_difficulty_map.keys())]
        else:
            # Find closest lower episode
            diff = 0.0
            sorted_eps = sorted(episode_difficulty_map.keys())
            for e in sorted_eps:
                if e <= ep_num:
                    diff = episode_difficulty_map[e]
                else:
                    break
        difficulties.append(diff)
        print(f"Checkpoint {os.path.basename(ckpt)} -> Difficulty {diff:.2f}")

    if not agents:
        print("No agents loaded.")
        return

    # Simulation Loop
    # Reset all envs with FIXED SEED for consistency
    states = [e.reset(seed=seed)[0] for e in envs]
    
    # Video Writer
    target_fps = 50.0
    writer = None
    # Use the first env for dt (all same)
    env_dt = envs[0].dt
    
    if save_mp4:
        # Auto-generate filename if default
        if output_mp4 == "overlay.mp4":
             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
             output_mp4 = f"docs/images/overlay_montage_{timestamp}.mp4"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
        
        # Use 'avc1' for H.264 (widely supported) or 'mp4v' as fallback
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        width, height = 800, 600
        
        # Calculate stride for real-time playback
        # Sim FPS = 1/dt (e.g. 200)
        sim_fps = 1.0 / env_dt
        stride = max(1, int(sim_fps / target_fps))
        real_fps = sim_fps / stride
        
        writer = cv2.VideoWriter(output_mp4, fourcc, real_fps, (width, height))
        if not writer.isOpened():
             print("Failed to open video writer with avc1, trying mp4v")
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             writer = cv2.VideoWriter(output_mp4, fourcc, real_fps, (width, height))
    else:
        stride = max(1, int((1.0/env_dt) / target_fps))

    total_steps = int(duration / env_dt)
    
    print(f"Simulating for {duration}s ({total_steps} steps). Stride: {stride}")
    
    # Colors: Fade from Red (early) to Green (late)
    cmap = plt.get_cmap('viridis') # Blue to Green/Yellow
    colors = [cmap(i) for i in np.linspace(0, 1, len(agents))]
    # Convert to 0-255 RGB
    colors_rgb = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]

    frames_written = 0
    reward_fn_label = "Reward Fn: SwingUp + Balance"
    
    for step in range(total_steps):
        # Step each agent in its OWN physics universe
        rewards = []
        for i, agent in enumerate(agents):
            # Set physics for this agent
            envs[i].set_curriculum(difficulties[i])
            
            action, _ = agent.select_action(states[i], deterministic=True)
            scaled_action = action * envs[i].force_mag
            # Step Env
            next_state, reward, terminated, truncated, _ = envs[i].step(scaled_action)
            done = terminated or truncated
            
            if done:
                # Reset this agent's environment
                next_state, _ = envs[i].reset()
                # Note: We don't reset the 'step' counter for the video loop, 
                # but the agent starts a new episode.
                
            states[i] = next_state
            rewards.append(reward)
            
            # Render (Overlay)
        if step % stride == 0:
            # We pass the LAST state (Newest/Best) to clear screen and draw UI
            # This ensures the UI stats reflect the current best model
            # Note: visualizer.render updates the reward history for the agent passed to it.
            # Use the actual episode label for the newest agent
            newest_ckpt = selected_checkpoints[-1]
            try:
                newest_ep_label = os.path.basename(newest_ckpt).split('_')[-1].split('.')[0]
            except:
                newest_ep_label = "?"
                
            visualizer.render(states[-1], episode=newest_ep_label, reward=rewards[-1], reward_fn_label=reward_fn_label, seed=seed)
            
            # Draw older agents as ghosts
            for i in range(len(agents) - 1):
                # Manually draw pendulum with transparency
                visualizer.draw_pendulum(states[i], color_p1=colors_rgb[i], color_p2=colors_rgb[i], alpha=80)
            
            # Draw Legend (Bottom Left)
            legend_x = 10
            # Start from bottom
            legend_base_y = 550 
            
            # Draw in reverse order (Newest on top)
            for i in range(len(agents)-1, -1, -1):
                ckpt = selected_checkpoints[i]
                # Get Episode Number
                try:
                    ep_label = os.path.basename(ckpt).split('_')[-1].split('.')[0]
                except:
                    ep_label = "?"
                
                if i == len(agents) - 1:
                    # Newest Agent (Rendered as Blue/Red)
                    # Use Blue for text to match the primary pole color
                    color = (0, 0, 255) 
                    label_text = f"Ep {ep_label} (Final)"
                else:
                    color = colors_rgb[i]
                    label_text = f"Ep {ep_label}"

                # Draw Text with Shadow for readability
                
                # Shadow
                text_surf_shadow = visualizer.font.render(label_text, True, (0, 0, 0))
                visualizer.screen.blit(text_surf_shadow, (legend_x + 1, legend_base_y - i * 25 + 1))
                
                # Colored Text
                text_surf = visualizer.font.render(label_text, True, color)
                visualizer.screen.blit(text_surf, (legend_x, legend_base_y - i * 25))

            # Draw Info (Top Left) - Frame Counter
            visualizer.screen.blit(visualizer.font.render(f"Frame: {step}", True, (0, 0, 0)), (10, 200))
            
            # Capture frame
            if writer:
                # Convert Pygame surface to numpy array (RGB)
                view = pygame.surfarray.array3d(visualizer.screen)
                # Transpose to (Height, Width, Channels) and BGR for OpenCV
                view = view.transpose([1, 0, 2])
                view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                writer.write(view)
                frames_written += 1
                
    if writer:
        writer.release()
        print(f"Video saved to {output_mp4}")
        print(f"Total Frames Written: {frames_written}")
        print(f"Expected Duration: {frames_written / 50.0:.2f}s (at 50 FPS)")
        
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_checkpoints", type=int, default=5)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--save_mp4", action="store_true")
    parser.add_argument("--output_mp4", type=str, default="overlay.mp4")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consistency")
    args = parser.parse_args()
    
    # Note: We need to pass the seed to visualize_overlay, but the function signature currently doesn't accept it.
    # I need to update the function signature first.
    # Actually, I'll update the function signature in this same edit if possible, or just hack it here.
    # Better to update the signature.
    
    # Wait, I can't update the signature in this chunk easily because it's at the top of the file.
    # I will update the call here, and then update the signature in a separate edit.
    # Or I can just set the seed globally? No, better to pass it.
    
    # Let's assume I will update the signature in the next step.
    visualize_overlay(args.log_dir, args.num_checkpoints, args.duration, args.save_mp4, args.output_mp4, seed=args.seed)
