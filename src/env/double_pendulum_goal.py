r"""
Goal-conditioned double pendulum (Phase 6, experimental).

The observation is augmented with a one-hot target ID, and the reward is computed
relative to a per-target reference state. The dynamics inherit the corrected
gravity sign and curriculum from :class:`DoublePendulumCartEnv`.

Targets
-------
0: Down-Down (stable)
1: Up-Up     (unstable)
2: Down-Up   (unstable)
3: Up-Down   (unstable)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from src.env.double_pendulum import DoublePendulumCartEnv, angle_normalize


class DoublePendulumGoalEnv(DoublePendulumCartEnv):
    NUM_TARGETS = 4
    TARGETS = {
        0: (0.0, 0.0),
        1: (np.pi, np.pi),
        2: (0.0, np.pi),
        3: (np.pi, 0.0),
    }

    def __init__(self, *, render_mode: Optional[str] = None, wind_std: float = 0.0) -> None:
        super().__init__(render_mode=render_mode, wind_std=wind_std)

        # Augment obs with one-hot goal channel.
        base_high = self.observation_space.high
        high = np.concatenate([base_high, np.ones(self.NUM_TARGETS, dtype=np.float32)])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.target_mode: int = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        if options and "target_mode" in options:
            self.target_mode = int(options["target_mode"])
        else:
            self.target_mode = int(self.np_random.integers(0, self.NUM_TARGETS))
        return self._goal_obs(obs), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = super().step(action)
        # Override the reward with a goal-conditioned shaping term.
        x = float(self.state[0])
        theta1, theta2 = float(self.state[1]), float(self.state[2])
        t1_target, t2_target = self.TARGETS[self.target_mode]

        e1 = angle_normalize(theta1 - t1_target)
        e2 = angle_normalize(theta2 - t2_target)

        # Wide Gaussian shaping. Values picked to match the per-step magnitude of
        # ExponentialSwingUpReward at saturation (~150) so swap-in trains use
        # similar value-head scales.
        sigma = 0.5
        r_spatial = float(np.exp(-(e1 ** 2 + e2 ** 2) / (2.0 * sigma ** 2)))
        r_centring = float(np.exp(-(x ** 2) / (2.0 * 2.0 ** 2)))
        reward = 100.0 * r_spatial * r_centring

        return self._goal_obs(obs), reward, terminated, truncated, info

    def _goal_obs(self, base_obs: np.ndarray) -> np.ndarray:
        goal = np.zeros(self.NUM_TARGETS, dtype=np.float32)
        goal[self.target_mode] = 1.0
        return np.concatenate([base_obs, goal])
