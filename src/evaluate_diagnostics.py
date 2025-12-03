import torch
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent

class DiagnosticsEvaluator:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        
        # Load Env (Standard Physics for Evaluation)
        self.env = DoublePendulumCartEnv(reset_mode="down", wind_std=0.0)
        # Force standard difficulty for evaluation (Diff 1.0)
        self.env.set_curriculum(difficulty=1.0)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Load Agent
        self.agent = PPOAgent(self.state_dim, self.action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2, k_epochs=4)
        self.agent.load(model_path)
        self.agent.policy.eval() # Set to eval mode

    def evaluate(self, num_episodes=50, max_steps=2000):
        print(f"Evaluating model: {self.model_path}")
        print(f"Episodes: {num_episodes}, Max Steps: {max_steps}")
        
        metrics = {
            "success_rate_strict": [], # < 10 deg
            "success_rate_loose": [],  # < 20 deg
            "avg_reward": [],
            "avg_length": [],
            "steady_state_error_t1": [],
            "steady_state_error_t2": [],
            "control_effort": []
        }
        
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            ep_reward = 0
            steps = 0
            
            # Trajectory data
            t1_errors = []
            t2_errors = []
            forces = []
            
            for t in range(max_steps):
                # Deterministic Action
                action, _ = self.agent.select_action(state, noise_bias=np.zeros(self.action_dim))
                scaled_action = action * self.env.force_mag
                
                next_state, reward, terminated, truncated, _ = self.env.step(scaled_action)
                done = terminated or truncated
                
                # Metrics
                x, s1, c1, s2, c2, x_dot, t1_dot, t2_dot = next_state
                theta1 = np.arctan2(s1, c1)
                theta2 = np.arctan2(s2, c2)
                
                t1_err = abs((theta1 - np.pi + np.pi) % (2 * np.pi) - np.pi)
                t2_err = abs((theta2 - np.pi + np.pi) % (2 * np.pi) - np.pi)
                
                t1_errors.append(t1_err)
                t2_errors.append(t2_err)
                forces.append(abs(scaled_action[0]))
                
                ep_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Episode Metrics
            # Strict Success: Balanced (<10deg) for last 10% of episode?
            # Or just % of time upright?
            # Let's use % of time upright
            strict_upright_steps = np.sum((np.array(t1_errors) < 0.17) & (np.array(t2_errors) < 0.17))
            loose_upright_steps = np.sum((np.array(t1_errors) < 0.35) & (np.array(t2_errors) < 0.35))
            
            metrics["success_rate_strict"].append(strict_upright_steps / steps)
            metrics["success_rate_loose"].append(loose_upright_steps / steps)
            metrics["avg_reward"].append(ep_reward)
            metrics["avg_length"].append(steps)
            metrics["control_effort"].append(np.mean(forces))
            
            # Steady State Error (Last 20% of steps)
            if steps > 100:
                last_n = int(steps * 0.2)
                metrics["steady_state_error_t1"].append(np.mean(t1_errors[-last_n:]))
                metrics["steady_state_error_t2"].append(np.mean(t2_errors[-last_n:]))
            else:
                metrics["steady_state_error_t1"].append(np.mean(t1_errors))
                metrics["steady_state_error_t2"].append(np.mean(t2_errors))

        return metrics

    def generate_report(self, metrics, output_path):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        avg_strict = np.mean(metrics["success_rate_strict"]) * 100
        avg_loose = np.mean(metrics["success_rate_loose"]) * 100
        avg_reward = np.mean(metrics["avg_reward"])
        avg_len = np.mean(metrics["avg_length"])
        avg_sse_t1 = np.rad2deg(np.mean(metrics["steady_state_error_t1"]))
        avg_sse_t2 = np.rad2deg(np.mean(metrics["steady_state_error_t2"]))
        avg_effort = np.mean(metrics["control_effort"])
        
        report = f"""# Training Diagnostic Report
**Timestamp**: {timestamp}
**Model**: `{os.path.basename(self.model_path)}`

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **{avg_strict:.1f}%** | % Time with both poles < 10° |
| **Loose Success Rate** | {avg_loose:.1f}% | % Time with both poles < 20° |
| **Avg Reward** | {avg_reward:.0f} | Mean episode reward |
| **Avg Length** | {avg_len:.0f} | Mean episode steps (Max 2000) |
| **Steady State Error (P1)** | {avg_sse_t1:.2f}° | Avg error in last 20% of ep |
| **Steady State Error (P2)** | {avg_sse_t2:.2f}° | Avg error in last 20% of ep |
| **Control Effort** | {avg_effort:.2f} N | Avg force magnitude |

## Interpretation
*   **Strict Success**: >80% indicates solid stabilization.
*   **Steady State Error**: <5° is excellent, <10° is acceptable.
*   **Control Effort**: Lower is better (smoother control).

## Raw Metrics (First 5 Eps)
*   Rewards: {[f'{r:.0f}' for r in metrics['avg_reward'][:5]]}
*   Strict Success: {[f'{s*100:.1f}%' for s in metrics['success_rate_strict'][:5]]}
"""
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="docs/reports/latest_report.md", help="Output report path")
    parser.add_argument("--episodes", type=int, default=20, help="Number of eval episodes")
    args = parser.parse_args()
    
    evaluator = DiagnosticsEvaluator(args.model)
    metrics = evaluator.evaluate(num_episodes=args.episodes)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    evaluator.generate_report(metrics, args.output)
