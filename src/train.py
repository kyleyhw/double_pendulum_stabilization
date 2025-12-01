import torch
import numpy as np
from env.double_pendulum import DoublePendulumCartEnv
from agent.ppo import PPOAgent, Memory
import os
from datetime import datetime

import argparse

def train(load_model=None, max_episodes=5000):
    # Hyperparameters
    env_name = "DoublePendulumCart-v0"
    max_timesteps = 2000       # Max timesteps per episode (Increased for Swing-Up)
    update_timestep = 2000     # Update policy every n timesteps
    lr = 0.0003
    gamma = 0.99
    k_epochs = 4
    eps_clip = 0.2
    
    # Logging
    log_interval = 20          # Print avg reward every n episodes
    save_interval = 200        # Save model every n episodes
    
    # Create Env
    # Swing-Up Task: Start "down", no wind for now.
    env = DoublePendulumCartEnv(reset_mode="down", wind_std=0.0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create Agent
    ppo = PPOAgent(state_dim, action_dim, lr, gamma, eps_clip, k_epochs)
    
    if load_model:
        print(f"Loading model from {load_model}...")
        ppo.load(load_model)
        
    memory = Memory()
    
    # Logging Setup
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV Logging
    log_file = os.path.join(log_dir, f"training_log_{run_name}.csv")
    with open(log_file, "w") as f:
        f.write("episode,reward,length\n")

    print(f"Starting training: {run_name}")
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
    
    # Training Loop
    time_step = 0
    running_reward = 0
    avg_length = 0
    
    try:
        # Curriculum Schedule
        # Anneal alpha from 0.0 to 1.0 over 'curriculum_episodes'
        curriculum_episodes = 3000
        
        for i_episode in range(1, max_episodes + 1):
            # Calculate alpha
            if i_episode <= curriculum_episodes:
                alpha = (i_episode - 1) / curriculum_episodes
            else:
                alpha = 1.0
                
            # Set Curriculum
            env.set_curriculum(alpha)
            
            state, _ = env.reset()
            current_ep_reward = 0
            
            for t in range(max_timesteps):
                time_step += 1
                
                # Select Action
                action, log_prob = ppo.select_action(state)
                
                # Step Env
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Save to memory
                memory.states.append(state)
                memory.actions.append(action)
                memory.log_probs.append(log_prob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                state = next_state
                current_ep_reward += reward
                
                # Update PPO
                if time_step % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear()
                    time_step = 0
                
                if done:
                    break
            
            running_reward += current_ep_reward
            avg_length += t
            
            # Log to CSV
            with open(log_file, "a") as f:
                f.write(f"{i_episode},{current_ep_reward},{t}\n")
            
            # Logging
            if i_episode % log_interval == 0:
                avg_reward = running_reward / log_interval
                avg_len = avg_length / log_interval
                # Renamed 'Avg Length' to 'Avg Lifetime' per user request
                # dt = 0.02s
                avg_lifetime_sec = avg_len * 0.02
                print(f"Episode {i_episode} \t Alpha: {alpha:.2f} \t Avg Reward: {avg_reward:.2f} \t Avg Lifetime: {avg_len:.2f} steps ({avg_lifetime_sec:.2f} s)")
                running_reward = 0
                avg_length = 0
                
            # Save Model
            if i_episode % save_interval == 0:
                save_path = os.path.join(log_dir, f"ppo_{run_name}_{i_episode}.pth")
                ppo.save(save_path)
                print(f"Model saved to {save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        save_path = os.path.join(log_dir, f"ppo_{run_name}_final.pth")
        ppo.save(save_path)
        print(f"Final model saved to {save_path}")
        print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Path to model to load")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    args = parser.parse_args()
    
    train(load_model=args.load, max_episodes=args.episodes)
