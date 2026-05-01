r"""
Run a trained PPO policy in the cart-pendulum environment, optionally
recording video. Loads the observation normaliser from the checkpoint so the
policy sees the same input distribution it was trained on.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pygame

# Add project root to path.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.ppo import PPOAgent  # noqa: E402
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.env.single_pendulum import SinglePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl, VelocityControl  # noqa: E402
from src.strategies.rewards import (  # noqa: E402
    DoublePendulumStandardReward,
    EnergyShapingReward,
    ExponentialSwingUpReward,
    HybridLQRSwingUpReward,
    LQRCostReward,
    SinglePendulumStandardReward,
)
from src.utils.normalize import NormalizeObservation  # noqa: E402
from src.utils.visualizer import Visualizer  # noqa: E402


def run_simulation(
    *,
    model_path: str | None = None,
    duration: float = 20.0,
    wind_std: float = 0.0,
    save_mp4: bool = False,
    output_mp4: str | None = None,
    reset_mode: str = "down",
    headless: bool = False,
    episode_label: str = "Final",
    difficulty: float = 1.0,
    reward_fn_label: str = "Reward Fn: Exponential SwingUp",
    seed: int = 42,
    env_name: str = "double",
    control: str = "velocity",
    reward_kind: str = "exponential",
) -> None:
    control_strategy = ForceControl() if control == "force" else VelocityControl()
    reward_strategy = {
        "exponential": ExponentialSwingUpReward(),
        "energy": EnergyShapingReward(),
        "lqr": LQRCostReward(),
        "hybrid": HybridLQRSwingUpReward(),
        "standard": (SinglePendulumStandardReward() if env_name == "single"
                     else DoublePendulumStandardReward()),
    }[reward_kind]

    EnvCls = SinglePendulumCartEnv if env_name == "single" else DoublePendulumCartEnv
    raw_env = EnvCls(
        wind_std=wind_std,
        reset_mode=reset_mode,
        control_strategy=control_strategy,
        reward_strategy=reward_strategy,
    )
    if wind_std > 0:
        raw_env.set_wind_pinned(wind_std)
    raw_env.set_curriculum(difficulty)
    if wind_std > 0:
        raw_env.set_wind_pinned(wind_std)

    env = NormalizeObservation(raw_env, training=False)
    viz = Visualizer(raw_env, headless=headless)

    state_dim = env.observation_space.shape[0]
    action_dim = raw_env.action_space.shape[0]

    agent: PPOAgent | None = None
    if model_path:
        print(f"[load] {model_path}")
        agent = PPOAgent(state_dim, action_dim)
        payload = agent.load(model_path)
        if "obs_rms" in payload:
            env.obs_rms.load_state_dict(payload["obs_rms"])
        else:
            print("[warn] no obs_rms in checkpoint — running with identity normaliser.")
        agent.policy.eval()
    else:
        print("[run] random actions (no model).")

    obs, _ = env.reset(seed=seed, options={"mode": reset_mode})

    target_fps = 50.0
    stride = max(1, int((1.0 / raw_env.dt) / target_fps))
    real_fps = (1.0 / raw_env.dt) / stride

    video_writer = None
    if save_mp4:
        import cv2
        if output_mp4 is None or output_mp4.endswith("final_run.mp4"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_mp4 = f"docs/images/final_run_{ts}.mp4"
        os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_mp4, fourcc, real_fps, (800, 600))
        print(f"[rec]  {output_mp4} @ {real_fps:.1f} fps (stride {stride})")

    max_steps = int(duration / raw_env.dt) if duration > 0 else 10**9
    print("Controls: LEFT/RIGHT arrow keys to push the cart manually.")

    try:
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if agent:
                action, _ = agent.select_action(obs, deterministic=True)
            else:
                keys = pygame.key.get_pressed()
                vel_cmd = (-1.0 if keys[pygame.K_LEFT] else 0.0) + \
                          (1.0 if keys[pygame.K_RIGHT] else 0.0)
                action = np.array([vel_cmd], dtype=np.float32)

            obs, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)

            if not headless or (save_mp4 and step % stride == 0):
                viz.render(raw_env.state, force=float(action[0]),
                           episode=episode_label, step=step, reward=float(reward),
                           reward_fn_label=reward_fn_label, seed=seed)

            if save_mp4 and video_writer is not None and step % stride == 0:
                import cv2
                frame = viz.get_frame()
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if done:
                obs, _ = env.reset(options={"mode": reset_mode})

            if not headless and not save_mp4:
                time.sleep(raw_env.dt)
    except KeyboardInterrupt:
        print("[interrupt]")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[saved] {output_mp4}")
        viz.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cart-pendulum simulation.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--save_mp4", action="store_true")
    parser.add_argument("--output", type=str, default="docs/images/final_run.mp4")
    parser.add_argument("--reset_mode", type=str, default="down")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--env", type=str, default="double", choices=["single", "double"])
    parser.add_argument("--control", type=str, default="velocity",
                        choices=["force", "velocity"])
    parser.add_argument("--reward", type=str, default="exponential",
                        choices=["standard", "exponential", "energy", "lqr", "hybrid"])
    args = parser.parse_args()

    run_simulation(
        model_path=args.model, duration=args.duration, wind_std=args.wind,
        save_mp4=args.save_mp4, output_mp4=args.output, reset_mode=args.reset_mode,
        headless=args.headless, seed=args.seed, difficulty=args.difficulty,
        env_name=args.env, control=args.control, reward_kind=args.reward,
    )
