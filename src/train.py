import torch
import numpy as np
import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent, Memory

def train(load_model=None, max_episodes=5000):
    # Hyperparameters
    env_name = "DoublePendulumCart-v0"
    max_timesteps = 4000       # Max timesteps per episode (20s at 200Hz)
    update_timestep = 4000     # Update policy every n timesteps
    lr = 0.0003
    gamma = 0.999              # Discount factor (Increased for 200Hz: Horizon ~5s)
    k_epochs = 4
    eps_clip = 0.1             # Clip parameter (Reduced for stability)
    
    # Logging
    log_interval = 20          # Print avg reward every n episodes
    save_interval = 200        # Save model every n episodes
    
    # Create Env
    # Swing-Up Task: Start "down" for Curriculum Learning
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
        f.write("Episode,Reward,Length,Difficulty,G,Friction,Threshold\n")
        
    print(f"Starting training: {run_name}")
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
    
    # Initialize Exploration Noise (OU Process)
    from src.utils.noise import OUNoise
    ou_noise = OUNoise(action_dim, theta=0.15, sigma=0.1)
    
    # Training Loop
    time_step = 0
    running_reward = 0
    avg_length = 0
    
    # Metrics
    running_success_rate = 0
    running_top_kinetic = 0
    running_cart_vel_std = 0
    
    # Curriculum
    difficulty = 0.0
    best_avg_reward = -float('inf') # Initialize for Slow Ratchet
    env.set_curriculum(difficulty)
    
    try:
        for i_episode in range(1, max_episodes+1):
            state, _ = env.reset()
            ou_noise.reset()
            current_ep_reward = 0
            
            # Metrics per episode
            ep_success_steps = 0
            ep_top_kinetic_sum = 0
            ep_top_kinetic_count = 0
            ep_cart_vels = []
            
            for t in range(max_timesteps):
                time_step += 1
                
                # Select Action
                noise = ou_noise.sample()
                action, log_prob = ppo.select_action(state, noise_bias=noise)
                
                # Scale Action
                scaled_action = action * env.force_mag
                
                # Step Env
                next_state, reward, terminated, truncated, _ = env.step(scaled_action)
                done = terminated or truncated
                
                # --- Metric Calculation ---
                x, s1, c1, s2, c2, x_dot, theta1_dot, theta2_dot = next_state
                
                # Reconstruct angles
                theta1 = np.arctan2(s1, c1)
                theta2 = np.arctan2(s2, c2)
                
                # Success (Upright < 10 deg)
                def angle_dist(th, target):
                    diff = (th - target + np.pi) % (2 * np.pi) - np.pi
                    return abs(diff)
                
                t1_err = angle_dist(theta1, np.pi)
                t2_err = angle_dist(theta2, np.pi)
                
                is_upright = (t1_err < 0.17) and (t2_err < 0.17)
                if is_upright:
                    ep_success_steps += 1
                    ep_top_kinetic_sum += (theta1_dot**2 + theta2_dot**2)
                    ep_top_kinetic_count += 1
                
                ep_cart_vels.append(x_dot)
                
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
            
            # Update Running Metrics
            ep_success_rate = ep_success_steps / t if t > 0 else 0
            running_success_rate += ep_success_rate
            
            ep_avg_top_k = ep_top_kinetic_sum / ep_top_kinetic_count if ep_top_kinetic_count > 0 else 0.0
            running_top_kinetic += ep_avg_top_k
            
            ep_cart_std = np.std(ep_cart_vels) if len(ep_cart_vels) > 0 else 0.0
            running_cart_vel_std += ep_cart_std
            
            # Log to CSV
            with open(log_file, "a") as f:
                f.write(f"{i_episode},{current_ep_reward},{t},{difficulty},{env.g},{env.friction_cart},{np.rad2deg(env.reward_threshold)}\n")
            
            # Logging & Curriculum Update
            if i_episode % log_interval == 0:
                avg_reward = running_reward / log_interval
                avg_len = avg_length / log_interval
                
                # Metrics
                avg_success = running_success_rate / log_interval
                avg_top_k = running_top_kinetic / log_interval
                avg_cart_std = running_cart_vel_std / log_interval
                
                # --- Adaptive Curriculum (Slow Ratchet) ---
                # User Request: "1% at a time", "higher than *all previous* Rs"
                # User Request: "cap for if we actually find a stable solution"
                
                # Calculate Max Theoretical Reward (Exponential)
                # R = sum(exp(t*dt) - 1) for t in 1..max_timesteps
                # This is a constant for a given max_timesteps
                if 'max_theoretical_reward' not in locals():
                    steps = np.arange(1, max_timesteps + 1)
                    times = steps * env.dt
                    max_theoretical_reward = np.sum(np.exp(times) - 1.0)
                
                # Check for improvement (All-time high)
                improved = avg_reward > best_avg_reward
                
                # Check for Saturation (95% of Max Theoretical)
                # If we are near perfect, we MUST level up or finish.
                saturated = avg_reward > (0.95 * max_theoretical_reward)
                
                if improved or saturated:
                    # Increase difficulty
                    difficulty += 0.01
                    difficulty = min(difficulty, 1.0)
                    params = env.set_curriculum(difficulty)
                    print(f"*** Level Up! Difficulty: {difficulty:.2f} | G: {params['g']:.1f} | F: {params['friction_cart']:.2f} | Thresh: {params['reward_threshold_deg']:.1f} ***")
                    
                    # Update best (Ratchet)
                    best_avg_reward = avg_reward
                    
                    # Termination Condition
                    if difficulty >= 1.0 and saturated:
                        print(f"*** SOLVED! Difficulty 1.0 reached with Reward {avg_reward:.0f}/{max_theoretical_reward:.0f} ***")
                        torch.save(agent.policy.state_dict(), os.path.join(log_dir, f"ppo_{run_id}_final.pth"))
                        break
                
                print(f"Ep {i_episode} | Diff: {difficulty:.2f} | R: {avg_reward:.0f}/{max_theoretical_reward:.0f} | L: {avg_len:.0f} | Succ: {avg_success*100:.1f}% | BestR: {best_avg_reward:.0f}")
                
                running_reward = 0
                avg_length = 0
                running_success_rate = 0
                running_top_kinetic = 0
                running_cart_vel_std = 0
                
            # Save Model
            if i_episode % save_interval == 0:
                save_path = os.path.join(log_dir, f"ppo_{run_name}_{i_episode}.pth")
                ppo.save(save_path)
                print(f"Model saved to {save_path}")
        print("\nTraining interrupted by user.")
    finally:
        save_path = os.path.join(log_dir, f"ppo_{run_name}_final.pth")
        ppo.save(save_path)
        print(f"Final model saved to {save_path}")
        print("Training complete.")
        
        # --- Automated Visualization ---
        print("\nGenerating Post-Training Visualizations...")
        
        # 1. Learning Curve
        try:
            # Fix import for when running as script
            try:
                from plot_learning_curve import plot_learning_curve
            except ImportError:
                from src.plot_learning_curve import plot_learning_curve
                
            print("Generating Learning Curve...")
            plot_learning_curve(log_dir=log_dir, output="docs/images/learning_curve.png")
        except Exception as e:
            print(f"Failed to generate learning curve: {e}")
            
        # 2. Final Run Video
        try:
            try:
                from simulate import run_simulation
            except ImportError:
                from src.simulate import run_simulation
                
            print("Generating Final Run Video...")
            run_simulation(model_path=save_path, duration=30.0, save_mp4=True, output="docs/images/final_run.mp4")
        except Exception as e:
            print(f"Failed to generate final run video: {e}")
            
        # 3. Overlay Montage
        try:
            try:
                from visualize_overlay import visualize_overlay
            except ImportError:
                from src.visualize_overlay import visualize_overlay
                
            print("Generating Overlay Montage...")
            visualize_overlay(log_dir=log_dir, num_checkpoints=5, duration=20.0, save_mp4=True, output_mp4="docs/images/overlay_montage.mp4")
        except Exception as e:
            print(f"Failed to generate overlay montage: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Path to model to load")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    args = parser.parse_args()
    
    train(load_model=args.load, max_episodes=args.episodes)
