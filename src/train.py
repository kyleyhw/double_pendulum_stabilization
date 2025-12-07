import torch
import numpy as np
import sys
import os
import argparse
from datetime import datetime
import collections
from typing import Tuple, Dict, Any, Optional, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.env.single_pendulum import SinglePendulumCartEnv
from src.agent.ppo import PPOAgent, Memory
from src.utils.noise import OUNoise
from src.strategies.controls import ForceControl, VelocityControl
from src.strategies.rewards import (
    SwingUpBalanceReward, 
    DoublePendulumStandardReward, 
    SinglePendulumStandardReward,
    ExponentialSwingUpReward
)

def get_env_and_strategies(args):
    """Factory function to create Env and Strategies based on args."""
    
    # 1. Control Strategy
    if args.control == "force":
        control_strategy = ForceControl()
    elif args.control == "velocity":
        control_strategy = VelocityControl()
    else:
        raise ValueError(f"Unknown control strategy: {args.control}")
        
    # 2. Reward Strategy
    if args.reward == "exponential":
        reward_strategy = ExponentialSwingUpReward()
    elif args.reward == "standard":
        if args.env == "single":
            reward_strategy = SinglePendulumStandardReward()
        else:
            reward_strategy = DoublePendulumStandardReward()
    else:
        raise ValueError(f"Unknown reward strategy: {args.reward}")
        
    # 3. Environment
    if args.env == "single":
        env = SinglePendulumCartEnv(
            reset_mode="down", 
            control_strategy=control_strategy,
            reward_strategy=reward_strategy
        )
        env_name = "SinglePendulumCart-v0"
    elif args.env == "double":
        env = DoublePendulumCartEnv(
            reset_mode="down", 
            control_strategy=control_strategy,
            reward_strategy=reward_strategy
        )
        env_name = "DoublePendulumCart-v0"
    else:
        raise ValueError(f"Unknown environment: {args.env}")
        
    return env, env_name, reward_strategy

def train(args):
    # Set Seed
    seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Training with Seed: {seed}")
    print(f"Configuration: Env={args.env}, Control={args.control}, Reward={args.reward}")
    
    # Create Env & Strategies
    env, env_name, reward_strategy = get_env_and_strategies(args)
    
    # Hyperparameters
    max_timesteps = 4000       # 20s at 200Hz
    update_timestep = 4000     
    lr = 0.0003
    gamma = 0.999              
    k_epochs = 4
    eps_clip = 0.1             
    
    # Logging
    log_interval = 50          
    save_interval = 200        
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create Agent
    ppo = PPOAgent(state_dim, action_dim, lr, gamma, eps_clip, k_epochs)
    
    if args.load:
        print(f"Loading model from {args.load}...")
        ppo.load(args.load)
        
    memory = Memory()
    ou_noise = OUNoise(action_dim, theta=0.5, sigma=0.3)
    
    # Logging Setup
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env}_{args.control}_{args.reward}_{timestamp}"
    print(f"Run Name: {run_name}")
    
    log_file = os.path.join(log_dir, f"training_log_{run_name}.csv")
    
    with open(log_file, "w") as f:
        f.write("Episode,Reward,Length,Difficulty,G,Friction,Threshold_Deg\n")
        
    # Training Loop State
    time_step = 0
    running_reward = 0
    running_time_above = 0
    avg_length = 0
    
    # Curriculum State
    difficulty = args.start_difficulty
    best_avg_reward = -float('inf')
    reward_window = collections.deque(maxlen=log_interval)
    env.set_curriculum(difficulty)
    episodes_since_last_levelup = 0
    
    # Pre-calculate Max Reward (Only strictly valid for Exponential)
    # For Standard, we will estimate or just track progress.
    steps = np.arange(1, max_timesteps + 1)
    if isinstance(reward_strategy, ExponentialSwingUpReward):
        max_theoretical_reward = np.sum(np.exp(steps * env.dt) - 1.0)
    else:
        # Standard: Max 1.0 per step (approx)
        max_theoretical_reward = max_timesteps * 1.5 # 1.5 max with centering bonus
    
    print(f"Max Theoretical Reward: {max_theoretical_reward:.1f}")

    try:
        for i_episode in range(1, args.episodes+1):
            state, _ = env.reset()
            ou_noise.reset()
            current_ep_reward = 0
            ep_steps_above_threshold = 0
            
            for t in range(max_timesteps):
                time_step += 1
                
                # Stagnation Jiggle
                jiggle_std = min(1.0, episodes_since_last_levelup * 0.0005)
                
                noise = ou_noise.sample()
                action, log_prob = ppo.select_action(state, noise_bias=noise, min_std=jiggle_std)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # --- Metric Calculation ---
                # Parse Observation Correctly: [x, sin, cos, ...]
                # Reconstruct theta from sin(t), cos(t)
                angles_up = []
                
                if args.env == "double":
                    # Obs: [x, s1, s2, c1, c2, xd, t1d, t2d]
                    # Indices: s1=1, s2=2, c1=3, c2=4
                    s1 = next_state[1]
                    s2 = next_state[2]
                    c1 = next_state[3]
                    c2 = next_state[4]
                    t1_val = np.arctan2(s1, c1)
                    t2_val = np.arctan2(s2, c2)
                    angles_up = [t1_val, t2_val]
                else:
                    # Single Obs: [x, s, c, xd, td]
                    s1, c1 = next_state[1], next_state[2]
                    t1_val = np.arctan2(s1, c1)
                    angles_up = [t1_val]
                
                all_up = True
                for theta in angles_up:
                    # Normalized error around PI (Up)
                    # Note: We reconstructed theta from sin/cos, so it's in [-pi, pi].
                    # Up is PI.
                    err = abs(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi)))
                    if err >= env.reward_strategy.reward_threshold:
                        all_up = False
                        break
                
                if all_up: 
                    ep_steps_above_threshold += 1
                
                # Memory
                memory.states.append(state)
                memory.actions.append(action)
                memory.log_probs.append(log_prob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                state = next_state
                current_ep_reward += reward
                
                if time_step % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear()
                    time_step = 0
                
                if done:
                    break
            
            # Post-Episode Updates
            running_reward += current_ep_reward
            running_time_above += (ep_steps_above_threshold / max_timesteps)
            avg_length += (t + 1)
            
            reward_window.append(current_ep_reward)
            episodes_since_last_levelup += 1
            
            # --- Ratchet Logic ---
            if len(reward_window) >= log_interval:
                window_avg_reward = np.mean(reward_window)
                
                if window_avg_reward > best_avg_reward:
                    best_avg_reward = window_avg_reward
                    
                saturated = window_avg_reward > (0.95 * max_theoretical_reward)
                
                # Difficulty Gate: Time Above Threshold
                required_time_above = difficulty * 0.90
                window_time_above = running_time_above / log_interval
                
                if (window_time_above > required_time_above) and (window_avg_reward >= best_avg_reward):
                    difficulty += 0.01
                    difficulty = min(difficulty, 1.0)
                    episodes_since_last_levelup = 0
                    params = env.set_curriculum(difficulty)
                    print(f"\n*** Level Up! Difficulty: {difficulty:.2f} | TimeAbove: {window_time_above*100:.1f}% | Rewards: {window_avg_reward:.0f} ***")

                    if difficulty >= 1.0 and saturated:
                         print(f"\n*** SOLVED! Reward {window_avg_reward:.0f} ***")
                         torch.save(ppo.policy.state_dict(), os.path.join(log_dir, f"ppo_{run_name}_final.pth"))
                         break

            # Log CSV
            with open(log_file, "a") as f:
                # G and Friction might not vary in Single Pendulum env the same way? 
                # SinglePendulum also has same set_curriculum signature.
                f.write(f"{i_episode},{current_ep_reward},{t+1},{difficulty},{env.g},{env.friction_cart},{np.rad2deg(env.reward_strategy.reward_threshold)}\n")

            # Print Stats
            if i_episode % log_interval == 0:
                avg_r = running_reward / log_interval
                avg_l = avg_length / log_interval
                avg_time = running_time_above / log_interval
                
                print(f"Ep {i_episode} | Diff: {difficulty:.2f} | R: {avg_r:.0f} | Len: {avg_l:.0f} | Time: {avg_time*100:.1f}% | Best: {best_avg_reward:.0f}")
                
                running_reward = 0
                running_time_above = 0
                avg_length = 0
                
                if i_episode % save_interval == 0:
                     save_path = os.path.join(log_dir, f"ppo_{run_name}_{i_episode}.pth")
                     ppo.save(save_path)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        save_path = os.path.join(log_dir, f"ppo_{run_name}_final.pth")
        ppo.save(save_path)
        print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="double", choices=["single", "double"], help="Environment type")
    parser.add_argument("--control", type=str, default="velocity", choices=["force", "velocity"], help="Control strategy")
    parser.add_argument("--reward", type=str, default="exponential", choices=["standard", "exponential"], help="Reward strategy")
    
    parser.add_argument("--load", type=str, help="Path to model to load")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--start_difficulty", type=float, default=0.0, help="Starting difficulty (0.0 to 1.0)")
    
    args = parser.parse_args()
    train(args)
