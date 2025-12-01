import torch
import numpy as np
import sys
import os
import argparse
import argparse
import time
import pygame

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

def run_simulation(model_path=None, duration=20.0, wind_std=0.0, save_gif=False, output_gif="docs/images/final_run.gif"):
    """
    Runs the simulation loop with visualization.
    
    Args:
        model_path (str): Path to a saved PPO checkpoint. If None, uses random actions.
        duration (float): Duration to run in seconds.
        save_gif (bool): Whether to save the run as a GIF.
        output_gif (str): Path to save the GIF.
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

    state, _ = env.reset()
    step = 0
    start_time = time.time()
    frames = []
    
    print("Starting simulation...")
    print("Controls:")
    print("  LEFT ARROW : Apply force Left")
    print("  RIGHT ARROW: Apply force Right")
    print("  Close window to exit.")
    
    from PIL import Image
    
    try:
        while duration <= 0 or (time.time() - start_time < duration):
            # Select Action
            if agent:
                action, _ = agent.select_action(state)
            else:
                action = env.action_space.sample()
            
            # Handle User Input (Impulses)
            keys = pygame.key.get_pressed()
            impulse = 0.0
            if keys[pygame.K_LEFT]:
                impulse = -10.0 # Push Left
            elif keys[pygame.K_RIGHT]:
                impulse = 10.0 # Push Right
            
            if impulse != 0:
                env.apply_impulse(impulse)
            
            # Step Env
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Render
            viz.render(state, force=action[0], external_force=impulse, step=step, reward=reward)
            
            if save_gif:
                str_data = pygame.image.tostring(viz.screen, 'RGBA')
                img = Image.frombytes('RGBA', (viz.width, viz.height), str_data)
                frames.append(img)
            
            state = next_state
            step += 1
            
            if terminated or truncated:
                print(f"Episode finished at step {step}. Resetting.")
                if save_gif:
                    break # Stop recording after one episode
                state, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        viz.close()
        if save_gif and frames:
            print(f"Saving GIF to {output_gif}...")
            os.makedirs(os.path.dirname(output_gif), exist_ok=True)
            frames[0].save(
                output_gif,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=16, # ~60 fps
                loop=0
            )
            print("GIF saved.")
        print("Simulation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Double Pendulum Simulation")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 for infinite)")
    parser.add_argument("--wind", type=float, default=0.0, help="Standard deviation of wind force")
    parser.add_argument("--save_gif", action="store_true", help="Save the run as a GIF")
    parser.add_argument("--output", type=str, default="docs/images/final_run.gif", help="Output path for GIF")
    
    args = parser.parse_args()
    
    run_simulation(args.model, args.duration, args.wind, args.save_gif, args.output)
