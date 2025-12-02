import torch
import numpy as np
import sys
import os
import argparse
import time
import pygame

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

def run_simulation(model_path=None, duration=20.0, wind_std=0.0, save_mp4=False, output_mp4=None, reset_mode="down", headless=False, episode_label="Final", difficulty=1.0, reward_fn_label="Reward Fn: SwingUp + Balance", seed=42):
    """
    Runs the Double Pendulum Cart simulation.

    Args:
        model_path (str): Path to the trained PPO model checkpoint. If None, random actions are used.
        duration (float): Duration of the simulation in seconds. If 0, runs indefinitely.
        wind_std (float): Standard deviation of the wind force applied to the cart.
        save_mp4 (bool): Whether to save the run as an MP4 video.
        output_mp4 (str): Path to save the MP4. If None, generates a timestamped filename.
        reset_mode (str): Mode for resetting the environment (e.g., 'down', 'random').
        headless (bool): Run without opening a window (useful for background tasks).
        episode_label (str): Label to display for the episode number (default: "Final").
        difficulty (float): Curriculum difficulty level (0.0 to 1.0).
        reward_fn_label (str): Label for the reward function display.
    """
    env = DoublePendulumCartEnv(wind_std=wind_std, reset_mode=reset_mode)
    env.set_curriculum(difficulty)
    viz = Visualizer(env, headless=headless)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = None
    if model_path:
        print(f"Loading model from {model_path}...")
        agent = PPOAgent(state_dim, action_dim)
        agent.load(model_path)
        agent.policy.eval() # Set to eval mode
    else:
        print("No model provided. Running with random actions.")

    state, _ = env.reset(seed=seed, options={"mode": reset_mode})
    step = 0
    
    print("Starting simulation...")
    print("Controls:")
    print("  LEFT ARROW : Apply force Left")
    print("  RIGHT ARROW: Apply force Right")
    print("  Close window to exit.")
    
    # Video Saving Setup
    video_writer = None
    target_fps = 50.0
    stride = 1
    if save_mp4:
        import cv2
        from datetime import datetime
        
        # Auto-generate filename if not provided or if it's the default placeholder
        if output_mp4 is None or output_mp4 == "docs/images/final_run.mp4":
             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
             output_mp4 = f"docs/images/final_run_{timestamp}.mp4"
             
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
        
        # Calculate stride to match target FPS
        # Sim FPS = 1/dt (e.g. 200)
        # Stride = Sim FPS / Target FPS
        sim_fps = 1.0 / env.dt
        stride = max(1, int(sim_fps / target_fps))
        real_fps = sim_fps / stride
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
        # Pygame screen is (800, 600)
        video_writer = cv2.VideoWriter(output_mp4, fourcc, real_fps, (800, 600))
        print(f"Recording to {output_mp4} at {real_fps:.2f} FPS (Stride: {stride})...")

    # Calculate max steps
    max_steps = int(duration / env.dt) if duration > 0 else float('inf')

    try:
        while step < max_steps:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get Action
            if agent:
                action, _ = agent.select_action(state, deterministic=True)
                action = action * env.force_mag
            else:
                # Manual Control
                keys = pygame.key.get_pressed()
                force = 0.0
                if keys[pygame.K_LEFT]:
                    force = -10.0
                elif keys[pygame.K_RIGHT]:
                    force = 10.0
                action = np.array([force])
            
            # Step Env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Render
            # Only render if visible or capturing frame
            if not headless or (save_mp4 and step % stride == 0):
                viz.render(state, force=action[0], episode=episode_label, step=step, reward=reward, reward_fn_label=reward_fn_label, seed=seed)
            
            # Capture Frame
            if save_mp4 and video_writer is not None and step % stride == 0:
                frame = viz.get_frame()
                # Pygame is RGB, OpenCV is BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            state = next_state
            step += 1
            
            if done:
                print(f"Episode finished at step {step}. Resetting.")
                state, _ = env.reset(options={"mode": reset_mode})
                # Do NOT reset step count, as we want to capture 'duration' seconds of simulation
                # But we might want to reset 'step' if we want to restart the episode?
                # Actually, 'step' here is global simulation step.
                # If we want to run for 20s, we just keep incrementing step.
                
                # However, viz.render uses 'step' for frame count?
                # Yes.
                pass
                
            # Enforce real-time playback if not headless and not saving video (optional)
            if not headless and not save_mp4:
                # Simple sleep to prevent super-fast rendering
                time.sleep(env.dt)
                
    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {output_mp4}")
        viz.close()
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Double Pendulum Simulation")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 for infinite)")
    parser.add_argument("--wind", type=float, default=0.0, help="Standard deviation of wind force")
    parser.add_argument("--save_mp4", action="store_true", help="Save run as MP4")
    parser.add_argument("--output", type=str, default="docs/images/final_run.mp4", help="Output MP4 path")
    parser.add_argument("--reset_mode", type=str, default="down", help="Reset mode (up/down/random)")
    parser.add_argument("--headless", action="store_true", help="Run without window")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--difficulty", type=float, default=1.0, help="Curriculum difficulty (0.0 to 1.0)")
    args = parser.parse_args()
    
    run_simulation(model_path=args.model, duration=args.duration, wind_std=args.wind, save_mp4=args.save_mp4, output_mp4=args.output, reset_mode=args.reset_mode, headless=args.headless, seed=args.seed, difficulty=args.difficulty)
