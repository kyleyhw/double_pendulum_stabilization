import torch
import numpy as np
from env.double_pendulum import DoublePendulumCartEnv
from agent.ppo import PPOAgent, Memory
import os
from datetime import datetime

def train():
    # Hyperparameters
    env_name = "DoublePendulumCart-v0"
    max_episodes = 5000        # Max training episodes
    max_timesteps = 1000       # Max timesteps per episode
    update_timestep = 2000     # Update policy every n timesteps
    lr = 0.0003
    gamma = 0.99
    k_epochs = 4
    eps_clip = 0.2
    
    # Logging
    log_interval = 20          # Print avg reward every n episodes
    save_interval = 500        # Save model every n episodes
    
    # Create Env
    env = DoublePendulumCartEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create Agent
    ppo = PPOAgent(state_dim, action_dim, lr, gamma, eps_clip, k_epochs)
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
    
    for i_episode in range(1, max_episodes + 1):
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
            print(f"Episode {i_episode} \t Avg Reward: {avg_reward:.2f} \t Avg Length: {avg_len:.2f}")
            running_reward = 0
            avg_length = 0
            
        # Save Model
        if i_episode % save_interval == 0:
            save_path = os.path.join(log_dir, f"ppo_{run_name}_{i_episode}.pth")
            ppo.save(save_path)
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()
