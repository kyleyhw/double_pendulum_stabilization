import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from src.strategies.controls import ControlStrategy, VelocityControl
from src.strategies.rewards import RewardStrategy, ExponentialSwingUpReward

class SinglePendulumCartEnv(gym.Env):
    r"""
    Single Pendulum on a Cart Environment (Modular).
    
    System Description:
    -------------------
    A cart of mass $M$ moves along the x-axis. A single pole of mass $m$ and length $l$ is attached.
    
    State Vector $s \in \mathbb{R}^4$:
    $$ s = [x, \theta, \dot{x}, \dot{\theta}]^T $$
    
    Equations of Motion:
    --------------------
    $$ M(q) \ddot{q} + C(q, \dot{q}) + G(q) = F_{ext} $$
    
    Mass Matrix $M(q)$:
    $$ \begin{bmatrix} M+m & ml \cos\theta \\ ml \cos\theta & ml^2 \end{bmatrix} $$
    
    Coriolis Vector $C(q, \dot{q})$:
    $$ \begin{bmatrix} -ml \sin\theta \dot{\theta}^2 \\ 0 \end{bmatrix} $$
    
    Gravity Vector $G(q)$:
    $$ \begin{bmatrix} 0 \\ mgl \sin\theta \end{bmatrix} $$
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 render_mode: Optional[str] = None, 
                 wind_std: float = 0.0, 
                 reset_mode: str = "up",
                 control_strategy: Optional[ControlStrategy] = None,
                 reward_strategy: Optional[RewardStrategy] = None):
        super().__init__()
        
        # Strategies
        self.control_strategy = control_strategy or VelocityControl()
        self.reward_strategy = reward_strategy or ExponentialSwingUpReward()
        
        # System Parameters
        self.M = 1.0      # Mass of cart
        self.m = 0.5      # Mass of pole
        self.l = 1.0      # Length of pole
        self.g = 9.81     # Gravity
        
        self.dt = 0.005
        self.wind_std = wind_std
        self.reset_mode = reset_mode
        self.current_impulse = 0.0
        
        # Action Space (Delegated)
        self.action_space = self.control_strategy.get_action_space()
    
        # Observation Space
        high = np.array([
             5.0, 1.0, 1.0, np.inf, np.inf
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.state: np.ndarray = np.zeros(4, dtype=np.float32) # Initialize properly

        
        # Physics Curriculum
        self.friction_cart = 0.0
        self.friction_pole = 0.0

    def set_curriculum(self, difficulty: float):
        difficulty = np.clip(difficulty, 0.0, 1.0)
        
        # Physics Curriculum
        self.g = 2.0 + difficulty * (9.81 - 2.0)
        self.friction_cart = 0.5 * (1.0 - difficulty)
        self.friction_pole = 0.1 * (1.0 - difficulty)
        
        # Pass to Reward Strategy
        self.reward_strategy.set_curriculum(difficulty)
        
        return {
            "g": self.g,
            "friction_cart": self.friction_cart,
            "reward_difficulty": difficulty
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        
        mode = self.reset_mode
        if options and "mode" in options:
            mode = options["mode"]
            
        if mode == "up":
            self.state[1] += np.pi
        elif mode == "down":
            pass 
        elif mode == "random":
            self.state[1] = self.np_random.uniform(0, 2*np.pi)
            
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        x, theta, x_dot, theta_dot = self.state
        return np.array([
            x,
            np.sin(theta), np.cos(theta),
            x_dot, theta_dot
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Get Force from Strategy
        env_params = {
            'dt': self.dt,
            'velocity_index': 2, # x_dot is index 2 in Single Pendulum State [x, t, xd, td]
            'max_force': 5000.0
        }
        
        force = self.control_strategy.get_force(action, self.state, env_params)
        
        if self.wind_std > 0:
            force += self.np_random.normal(0, self.wind_std)
        force += self.current_impulse
        self.current_impulse = 0.0
        
        # 2. Integrate Physics
        self.state = self._rk4_step(self.state, force, self.dt)
        
        x = self.state[0]
        
        # 3. Terminate Logic (Position Only)
        # Use simple bounds check
        terminated = bool(
            x < -5.0
            or x > 5.0
        )
        truncated = False
        
        # 4. Reward from Strategy
        reward = self.reward_strategy.compute_reward(self.state, env_params)
        
        return self._get_obs(), reward, terminated, truncated, {}
        
    def apply_impulse(self, force: float):
        self.current_impulse = force

    def _rk4_step(self, state: np.ndarray, force: float, dt: float) -> np.ndarray:
        k1 = self._dynamics(state, force)
        k2 = self._dynamics(state + 0.5 * dt * k1, force)
        k3 = self._dynamics(state + 0.5 * dt * k2, force)
        k4 = self._dynamics(state + dt * k3, force)
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state

    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        r"""
        Computes accelerations $\ddot{q}$ for Single Pendulum.
        """
        x, theta, x_dot, theta_dot = state
        
        M, m = self.M, self.m
        l = self.l
        g = self.g
        
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Mass Matrix
        M_mat = np.array([
            [M + m, m * l * c],
            [m * l * c, m * l**2]
        ])
        
        # Damping
        damping_x = -self.friction_cart * x_dot
        damping_t = -self.friction_pole * theta_dot
        
        # Coriolis & Gravity
        # C term: -ml sin(t) t_dot^2
        # G term: mgl sin(t)
        
        C_vec = np.array([
            -m * l * s * theta_dot**2,
            0
        ])
        
        G_vec = np.array([
            0,
            m * g * l * s
        ])
        
        D_vec = np.array([damping_x, damping_t])
        F_vec = np.array([force, 0])
        
        # Solve M q_dd = F + D - C - G
        RHS = F_vec + D_vec - C_vec - G_vec
        q_dd = np.linalg.solve(M_mat, RHS)
        
        return np.concatenate(([x_dot, theta_dot], q_dd))
