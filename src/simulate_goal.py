import torch
import numpy as np
import sys
import os
import argparse
import time
import pygame

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum_goal import DoublePendulumGoalEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

def run_simulation(model_path=None, duration=0.0, wind_std=0.0, save_mp4=False, output_mp4="docs/images/goal_run.mp4"):
    """
    Runs the Goal-Conditioned Double Pendulum simulation.
    Interactive Keys:
    1: Down-Down
    2: Up-Up
    3: Down-Up
    4: Up-Down
    """
    env = DoublePendulumGoalEnv(wind_std=wind_std)
    viz = Visualizer(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = None
    if model_path:
        print(f"Loading model from {model_path}...")
        agent = PPOAgent(state_dim, action_dim)
        agent.load(model_path)
        agent.policy.eval()
    else:
        print("No model provided. Running with random actions.")

    # Start with Down-Down
    env.target_mode = 0
    state, _ = env.reset(options={"target_mode": 0})
    step = 0
    start_time = time.time()
    
    print("Starting simulation...")
    print("Controls:")
    print("  1: Target Down-Down")
    print("  2: Target Up-Up")
    print("  3: Target Down-Up")
    print("  4: Target Up-Down")
    print("  Close window to exit.")
    
    # Video Saving Setup
    video_writer = None
    if save_mp4:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_mp4, fourcc, 60.0, (800, 600))
        print(f"Recording to {output_mp4}...")

    try:
        while duration == 0 or (time.time() - start_time < duration):
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        env.target_mode = 0
                        print("Target: Down-Down")
                    elif event.key == pygame.K_2:
                        env.target_mode = 1
                        print("Target: Up-Up")
                    elif event.key == pygame.K_3:
                        env.target_mode = 2
                        print("Target: Down-Up")
                    elif event.key == pygame.K_4:
                        env.target_mode = 3
                        print("Target: Up-Down")

            # Get Action
            if agent:
                # Agent sees the augmented state (including new target)
                action, _ = agent.select_action(state)
            else:
                action = np.array([0.0])
            
            # Step Env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Render
            # We should visualize the target too.
            # Visualizer doesn't know about target_mode, but we can print it.
            target_names = ["Down-Down", "Up-Up", "Down-Up", "Up-Down"]
            target_name = target_names[env.target_mode]
            
            viz.render(state, force=action[0], episode=f"Goal: {target_name}", step=step, reward=reward)
            
            # Capture Frame
            if save_mp4 and video_writer is not None:
                frame = viz.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            state = next_state
            step += 1
            
            if done:
                print(f"Episode finished at step {step}. Resetting to same target.")
                state, _ = env.reset(options={"target_mode": env.target_mode})
                step = 0
                
    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {output_mp4}")
        viz.close()
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Goal-Conditioned Simulation")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 for infinite)")
    parser.add_argument("--wind", type=float, default=0.0, help="Standard deviation of wind force")
    parser.add_argument("--save_mp4", action="store_true", help="Save the run as an MP4 video")
    parser.add_argument("--output", type=str, default="docs/images/goal_run.mp4", help="Output path for MP4")
    
    args = parser.parse_args()
    
    run_simulation(args.model, args.duration, args.wind, args.save_mp4, args.output)
