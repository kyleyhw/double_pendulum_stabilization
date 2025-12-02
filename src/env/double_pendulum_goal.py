import numpy as np
from gymnasium import spaces
from src.env.double_pendulum import DoublePendulumCartEnv, angle_normalize

class DoublePendulumGoalEnv(DoublePendulumCartEnv):
    """
    Goal-Conditioned Double Pendulum Environment.
    
    Augments the observation space with a one-hot encoded target ID.
    The reward function dynamically adapts to the current target.
    
    Targets:
    0: Down-Down (Stable)
    1: Up-Up (Unstable)
    2: Down-Up (Unstable)
    3: Up-Down (Unstable)
    """
    def __init__(self, render_mode=None, wind_std=0.0):
        super().__init__(render_mode=render_mode, wind_std=wind_std)
        
        # 4 Goals -> One-Hot Vector of size 4
        # New Obs Dim: 6 (State) + 4 (Goal) = 10
        high = np.concatenate([
            self.observation_space.high,
            np.ones(4, dtype=np.float32)
        ])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.target_mode = 0 # Default to Down-Down
        
    def reset(self, seed=None, options=None):
        # Randomize target mode during training if specified, 
        # but for now let's default to random to learn all.
        super().reset(seed=seed)
        
        if options and 'target_mode' in options:
            self.target_mode = options['target_mode']
        else:
            # Random target
            self.target_mode = self.np_random.integers(0, 4)
            
        return self._get_obs(), {}
        
    def _get_obs(self):
        base_obs = self.state.astype(np.float32)
        
        # One-Hot Goal
        goal_obs = np.zeros(4, dtype=np.float32)
        goal_obs[self.target_mode] = 1.0
        
        return np.concatenate([base_obs, goal_obs])
        
    def step(self, action):
        # 1. Physics Step (Same as base)
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        if self.wind_std > 0:
            force += self.np_random.normal(0, self.wind_std)
        force += self.current_impulse
        self.current_impulse = 0.0
        self.state = self._rk4_step(self.state, force, self.dt)
        
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = self.state
        
        # Termination
        terminated = bool(
            x < -self.observation_space.high[0]
            or x > self.observation_space.high[0]
        )
        truncated = False
        
        # 2. Determine Target State based on Mode
        # Angles are relative to vertical down (0).
        # Down: 0, Up: pi
        
        if self.target_mode == 0: # Down-Down
            t1_target = 0.0
            t2_target = 0.0
        elif self.target_mode == 1: # Up-Up
            t1_target = np.pi
            t2_target = np.pi
        elif self.target_mode == 2: # Down-Up (Pole 1 Down, Pole 2 Up)
            t1_target = 0.0
            t2_target = np.pi
        elif self.target_mode == 3: # Up-Down (Pole 1 Up, Pole 2 Down)
            t1_target = np.pi
            t2_target = 0.0
            
        # 3. Calculate Errors
        t1_err = angle_normalize(theta1 - t1_target)
        t2_err = angle_normalize(theta2 - t2_target)
        
        # 4. Calculate Target Energy
        # V = -(m1+m2)g l1 cos(t1) - m2 g l2 cos(t2)
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        
        V_target = -(m1 + m2) * g * l1 * np.cos(t1_target) - m2 * g * l2 * np.cos(t2_target)
        # Target Kinetic is always 0 (Stabilization)
        E_target = V_target
        
        current_energy = self._get_energy()
        energy_diff = np.abs(current_energy - E_target)
        
        # 5. Reward Components (Reusing tuned weights)
        
        # Calculate current sigma based on curriculum
        # Linear interpolation: Wide -> Narrow
        current_sigma = (1.0 - self.curriculum_alpha) * self.sigma_start + self.curriculum_alpha * self.sigma_target

        # Spatial
        w1, w2, w3, w4 = 1.0, 1.0, 0.5, 0.1
        dist_sq = (
            w1 * t1_err**2 + 
            w2 * t2_err**2 + 
            w3 * x**2 + 
            w4 * (theta1_dot**2 + theta2_dot**2)
        )
        r_spatial = np.exp(-dist_sq / (current_sigma**2))
        
        # Energy
        # Scale energy difference? Energy can be large (~40J).
        energy_sigma = 10.0 * current_sigma
        r_energy = np.exp(-energy_diff / energy_sigma)
        
        # Kinetic Damping (Dual-Gaussian + Gating)
        # Recalculate V to get T
        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        V = -(m1 + m2) * g * l1 * c1 - m2 * g * l2 * c2
        T = current_energy - V
        
        r_kinetic_wide = np.exp(-T / (10.0 * current_sigma))
        r_kinetic_narrow = np.exp(-T / (1.0 * current_sigma))
        r_kinetic_raw = 0.2 * r_kinetic_wide + 0.8 * r_kinetic_narrow
        
        # Gating: Only apply kinetic reward when near SPATIAL target
        pos_dist_sq = t1_err**2 + t2_err**2
        gating = np.exp(-pos_dist_sq / (current_sigma**2))
        r_kinetic = r_kinetic_raw * gating
        
        # Combined
        w_spatial = 0.4
        w_energy = 0.3
        w_kinetic = 0.3
        
        reward = w_spatial * r_spatial + w_energy * r_energy + w_kinetic * r_kinetic
        
        return self._get_obs(), reward, terminated, truncated, {}
