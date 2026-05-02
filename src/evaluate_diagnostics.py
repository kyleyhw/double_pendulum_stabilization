r"""
Deterministic-policy evaluation of a trained checkpoint, producing a Markdown
report with success rates, steady-state error, and control effort.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.ppo import PPOAgent  # noqa: E402
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import VelocityControl  # noqa: E402
from src.strategies.rewards import ExponentialSwingUpReward  # noqa: E402
from src.utils.normalize import NormalizeObservation  # noqa: E402


class DiagnosticsEvaluator:
    def __init__(self, model_path: str, difficulty: float = 1.0) -> None:
        self.model_path = model_path
        self.difficulty = float(difficulty)
        raw = DoublePendulumCartEnv(
            reset_mode="down",
            control_strategy=VelocityControl(),
            reward_strategy=ExponentialSwingUpReward(),
            wind_std=0.0,
        )
        raw.set_curriculum(difficulty=self.difficulty)
        self.raw = raw
        self.env = NormalizeObservation(raw, training=False)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = raw.action_space.shape[0]
        self.agent = PPOAgent(self.state_dim, self.action_dim)
        payload = self.agent.load(model_path)
        if "obs_rms" in payload:
            self.env.obs_rms.load_state_dict(payload["obs_rms"])
        else:
            print("[warn] no obs_rms in checkpoint; results will be biased if normalisation matters.")
        self.agent.policy.eval()

    def evaluate(self, num_episodes: int = 50, max_steps: int = 4000) -> dict:
        print(f"[eval] {self.model_path}  episodes={num_episodes}  max_steps={max_steps}")
        metrics: dict[str, list] = {
            "success_rate_strict": [],
            "success_rate_loose": [],
            "avg_reward": [],
            "avg_length": [],
            "steady_state_error_t1": [],
            "steady_state_error_t2": [],
            "control_effort": [],
        }
        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=ep)
            ep_reward = 0.0
            t1_errs, t2_errs, forces = [], [], []
            for t in range(max_steps):
                action, _ = self.agent.select_action(obs, deterministic=True)
                obs, r, term, trunc, _ = self.env.step(action)
                ep_reward += float(r)

                t1, t2 = self.raw.state[1], self.raw.state[2]
                t1_errs.append(abs(np.arctan2(np.sin(t1 - np.pi), np.cos(t1 - np.pi))))
                t2_errs.append(abs(np.arctan2(np.sin(t2 - np.pi), np.cos(t2 - np.pi))))
                forces.append(abs(float(action[0])))
                if term or trunc:
                    break
            steps = t + 1
            t1a, t2a = np.array(t1_errs), np.array(t2_errs)
            metrics["success_rate_strict"].append(float(np.mean((t1a < 0.17) & (t2a < 0.17))))
            metrics["success_rate_loose"].append(float(np.mean((t1a < 0.35) & (t2a < 0.35))))
            metrics["avg_reward"].append(ep_reward)
            metrics["avg_length"].append(steps)
            metrics["control_effort"].append(float(np.mean(forces)))
            tail = max(1, steps // 5)
            metrics["steady_state_error_t1"].append(float(np.mean(t1a[-tail:])))
            metrics["steady_state_error_t2"].append(float(np.mean(t2a[-tail:])))
        return metrics

    def generate_report(self, metrics: dict, output_path: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_strict = float(np.mean(metrics["success_rate_strict"])) * 100
        avg_loose = float(np.mean(metrics["success_rate_loose"])) * 100
        avg_reward = float(np.mean(metrics["avg_reward"]))
        avg_len = float(np.mean(metrics["avg_length"]))
        sse_t1 = float(np.rad2deg(np.mean(metrics["steady_state_error_t1"])))
        sse_t2 = float(np.rad2deg(np.mean(metrics["steady_state_error_t2"])))
        eff = float(np.mean(metrics["control_effort"]))

        report = f"""# Training Diagnostic Report
**Timestamp**: {ts}
**Model**: `{os.path.basename(self.model_path)}`
**Evaluation difficulty (delta)**: {self.difficulty:.3f} (g = {self.raw.g:.3f}, mu_cart = {self.raw.friction_cart:.3f}, threshold = {np.rad2deg(self.raw.reward_strategy.reward_threshold):.1f} deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **{avg_strict:.1f}%** | % time both poles within 10° of upright |
| **Loose Success Rate** | {avg_loose:.1f}% | % time both poles within 20° of upright |
| **Avg Reward** | {avg_reward:.0f} | Mean episode reward |
| **Avg Length** | {avg_len:.0f} | Mean episode steps |
| **Steady-State Error (P1)** | {sse_t1:.2f}° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | {sse_t2:.2f}° | Avg error over the trailing 20% |
| **Control Effort** | {eff:.3f} | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[report] {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="docs/reports/latest_report.md")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="Curriculum delta at which to evaluate (use the value the "
                             "agent reached during training for an honest comparison).")
    args = parser.parse_args()

    ev = DiagnosticsEvaluator(args.model, difficulty=args.difficulty)
    m = ev.evaluate(num_episodes=args.episodes)
    ev.generate_report(m, args.output)
