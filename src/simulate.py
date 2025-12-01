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

def run_simulation(model_path=None, duration=20.0, wind_std=0.0, save_mp4=False, output_mp4="docs/images/final_run.mp4", reset_mode="down"):
    """
    Runs the Double Pendulum Cart simulation.

    Args:
        model_path (str): Path to the trained PPO model checkpoint. If None, random actions are used.
        duration (float): Duration of the simulation in seconds. If 0, runs indefinitely.
        wind_std (float): Standard deviation of the wind force applied to the cart.
        save_mp4 (bool): Whether to save the run as an MP4 video.
        output_mp4 (str): Path to save the MP4.
        reset_mode (str): Mode for resetting the environment (e.g., 'down', 'random').
    """
    env = DoublePendulumCartEnv(wind_std=wind_std)
    viz = Visualizer(env)
    
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

    state, _ = env.reset(options={"mode": reset_mode})
    step = 0
    start_time = time.time()
    
    print("Starting simulation...")
    print("Controls:")
    print("  LEFT ARROW : Apply force Left")
    print("  RIGHT ARROW: Apply force Right")
    print("  Close window to exit.")
    
    # Video Saving Setup
    video_writer = None
    if save_mp4:
        import cv2
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
        # Pygame screen is (800, 600)
        video_writer = cv2.VideoWriter(output_mp4, fourcc, 60.0, (800, 600))
        print(f"Recording to {output_mp4}...")

    try:
        while duration == 0 or (time.time() - start_time < duration):
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get Action
            if agent:
                action, _ = agent.select_action(state)
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
            viz.render(state, force=action[0], episode=1, step=step, reward=reward)
            
            # Capture Frame
            if save_mp4 and video_writer is not None:
                frame = viz.get_frame()
                # Pygame is RGB, OpenCV is BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            state = next_state
            step += 1
            
            if done:
                print(f"Episode finished at step {step}. Resetting.")
                state, _ = env.reset(options={"mode": reset_mode})
                step = 0
                # Don't break, just continue for duration
                
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
    parser.add_argument("--save_mp4", action="store_true", help="Save the run as an MP4 video")
    parser.add_argument("--output", type=str, default="docs/images/final_run.mp4", help="Output path for MP4")
    parser.add_argument("--reset_mode", type=str, default="down", help="Reset mode for the environment (e.g., 'down', 'random')")
    
    args = parser.parse_args()
    
    run_simulation(args.model, args.duration, args.wind, args.save_mp4, args.output, args.reset_mode)
