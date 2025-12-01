import torch
import numpy as np
import sys
import os
import argparse
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

def run_simulation(model_path=None, duration=20.0):
    """
    Runs the simulation loop with visualization.
    
    Args:
        model_path (str): Path to a saved PPO checkpoint. If None, uses random actions.
        duration (float): Duration to run in seconds.
    """
    env = DoublePendulumCartEnv()
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
    
    print("Starting simulation...")
    print("Press Close on the window to exit.")
    
    try:
        while time.time() - start_time < duration:
            # Select Action
            if agent:
                action, _ = agent.select_action(state)
            else:
                action = env.action_space.sample()
            
            # Step Env
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Render
            viz.render(state, force=action[0], step=step, reward=reward)
            
            state = next_state
            step += 1
            
            if terminated or truncated:
                print(f"Episode finished at step {step}. Resetting.")
                state, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        viz.close()
        print("Simulation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Double Pendulum Simulation")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    
    args = parser.parse_args()
    
    run_simulation(args.model, args.duration)
