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

def train(load_model=None, max_episodes=5000, seed=None, start_difficulty=0.0):
    # Set Seed
    if seed is None:
        seed = np.random.randint(0, 100000)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Training with Seed: {seed}")
    # Hyperparameters
    env_name = "DoublePendulumCart-v0"
    max_timesteps = 4000       # Max timesteps per episode (20s at 200Hz)
    update_timestep = 4000     # Update policy every n timesteps
    lr = 0.0003
    gamma = 0.999              # Discount factor (Increased for 200Hz: Horizon ~5s)
    k_epochs = 4
    eps_clip = 0.1             # Clip parameter (Reduced for stability)
    
    # Logging
    log_interval = 50          # Print avg reward every n episodes (Reduced for performance)
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
    log_file = os.path.join(log_dir, f"training_log_{run_name}.csv")
    
    # Initialize Log File
    with open(log_file, "w") as f:
        f.write("Episode,Reward,Length,Difficulty,G,Friction,Threshold_Deg\n")
        
    from src.utils.noise import OUNoise
    ou_noise = OUNoise(action_dim, theta=0.5, sigma=0.3)
    
    # Training Loop
    time_step = 0
    running_reward = 0
    avg_length = 0
    
    # Metrics
    running_time_above_threshold = 0
    running_top_kinetic = 0
    running_cart_vel_std = 0
    
    import collections
    
    # Curriculum
    difficulty = start_difficulty
    best_avg_reward = -float('inf') # Initialize for Slow Ratchet
    reward_window = collections.deque(maxlen=log_interval) # Sliding window for per-episode ratchet
    env.set_curriculum(difficulty)
    episodes_since_last_levelup = 0
    
    try:
        for i_episode in range(1, max_episodes+1):
            # print(".", end="", flush=True) # Progress indicator (Removed)
            state, _ = env.reset()
            ou_noise.reset()
            current_ep_reward = 0
            
            # Metrics per episode
            ep_steps_above_threshold = 0
            ep_top_kinetic_sum = 0
            ep_top_kinetic_count = 0
            ep_cart_vels = []
            
            for t in range(max_timesteps):
                time_step += 1
                
                # Select Action
                # Stagnation Jiggle: Increase exploration if stuck
                jiggle_std = min(1.0, episodes_since_last_levelup * 0.0005) # 0.0 at 0, 0.5 at 1000 eps
                
                noise = ou_noise.sample()
                action, log_prob = ppo.select_action(state, noise_bias=noise, min_std=jiggle_std)
                
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

                # Calculate errors for Curriculum Ratchet
                def angle_dist(th, target):
                    diff = (th - target + np.pi) % (2 * np.pi) - np.pi
                    return abs(diff)
                
                t1_err = angle_dist(theta1, np.pi)
                t2_err = angle_dist(theta2, np.pi)
                
                # Check if above CURRENT curriculum threshold
                is_above_threshold = (t1_err < env.reward_threshold) and (t2_err < env.reward_threshold)
                
                if is_above_threshold:
                    ep_steps_above_threshold += 1
                
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
            
            # Update Running Metric (Fraction of time above threshold)
            running_time_above_threshold += (ep_steps_above_threshold / max_timesteps)
            
            # Sliding Window for Ratchet
            reward_window.append(current_ep_reward)
            episodes_since_last_levelup += 1
            
            # --- Adaptive Curriculum (Per-Episode Ratchet) ---
            if len(reward_window) >= log_interval:
                window_avg_reward = np.mean(reward_window)
                
                # Calculate Max Theoretical Reward (Exponential)
                if 'max_theoretical_reward' not in locals():
                    steps = np.arange(1, max_timesteps + 1)
                    times = steps * env.dt
                    max_theoretical_reward = np.sum(np.exp(times) - 1.0)
                
                # Check for improvement (All-time high) - For UI only
                # Update Best Reward continuously so logs reflect reality
                if window_avg_reward > best_avg_reward:
                    best_avg_reward = window_avg_reward
                
                # Check for Saturation (95% of Max Theoretical)
                saturated = window_avg_reward > (0.95 * max_theoretical_reward)
                
                # DYNAMIC MASTERY GATE:
                # The required time above threshold scales with difficulty (0% -> 90%)
                required_time_above = difficulty * 0.90
                
                window_time_above = running_time_above_threshold / log_interval
                
                # Ratchet Condition: Advance if Time Above Threshold met AND Reward is All-Time High
                # User Requirement: "Train with Slow Ratchet (1% Increment, All-Time High)"
                # We use >= because we just updated best_avg_reward above.
                if (window_time_above > required_time_above) and (window_avg_reward >= best_avg_reward):
                    # Increase difficulty
                    difficulty += 0.01
                    difficulty = min(difficulty, 1.0)
                    episodes_since_last_levelup = 0 # Reset Stagnation Counter
                    params = env.set_curriculum(difficulty)
                    print(f"\n*** Level Up! Difficulty: {difficulty:.2f} | TimeAbove: {window_time_above*100:.1f}% (Req: {required_time_above*100:.0f}%) | G: {params['g']:.1f} | F: {params['friction_cart']:.2f} | Thresh: {params['reward_threshold_deg']:.1f} ***")
                    
                    # Termination Condition
                    if difficulty >= 1.0 and saturated:
                        print(f"\n*** SOLVED! Difficulty 1.0 reached with Reward {window_avg_reward:.0f} ***")
                        torch.save(ppo.policy.state_dict(), os.path.join(log_dir, f"ppo_{run_name}_final.pth"))
                        break

            # Log to CSV
            with open(log_file, "a") as f:
                f.write(f"{i_episode},{current_ep_reward},{t},{difficulty},{env.g},{env.friction_cart},{np.rad2deg(env.reward_threshold)}\n")
            
            # Logging (Print Stats)
            if i_episode % log_interval == 0:
                avg_reward = running_reward / log_interval
                avg_len = avg_length / log_interval
                
                # Metrics
                avg_time_above = running_time_above_threshold / log_interval
                
                print(f"\nEp {i_episode} | Diff: {difficulty:.2f} | R: {avg_reward:.0f} | L: {avg_len:.0f} | TimeAbove: {avg_time_above*100:.1f}% | BestR: {best_avg_reward:.0f} | Jiggle: {jiggle_std:.2f}")
                
                if avg_reward > 0.9 * max_theoretical_reward:
                    print(f"*** NEARLY SOLVED (90% Max Reward: {max_theoretical_reward:.0f}) ***")
                
                running_reward = 0
                avg_length = 0
                running_time_above_threshold = 0
                
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
            run_simulation(model_path=save_path, duration=30.0, save_mp4=True, output_mp4="docs/images/final_run.mp4", seed=seed)
        except Exception as e:
            print(f"Failed to generate final run video: {e}")
            
        # 3. Overlay Montage
        try:
            try:
                from visualize_overlay import visualize_overlay
            except ImportError:
                from src.visualize_overlay import visualize_overlay
                
            print("Generating Overlay Montage...")
            visualize_overlay(log_dir=log_dir, num_checkpoints=5, duration=20.0, save_mp4=True, output_mp4="docs/images/overlay_montage.mp4", seed=seed)
        except Exception as e:
            print(f"Failed to generate overlay montage: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Path to model to load")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--start_difficulty", type=float, default=0.0, help="Starting difficulty (0.0 to 1.0)")
    args = parser.parse_args()
    
    train(load_model=args.load, max_episodes=args.episodes, seed=args.seed, start_difficulty=args.start_difficulty)
