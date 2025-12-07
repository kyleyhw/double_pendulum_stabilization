import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from src.strategies.controls import ControlStrategy, VelocityControl
from src.strategies.rewards import RewardStrategy, ExponentialSwingUpReward

class DoublePendulumCartEnv(gym.Env):
    r"""
    Double Pendulum on a Cart Environment (Modular).
    
    System Description:
    -------------------
    A cart of mass $M$ moves along the x-axis. A pole of mass $m_1$ and length $l_1$ is attached to the cart.
    A second pole of mass $m_2$ and length $l_2$ is attached to the end of the first pole.
    
    State Vector $s \in \mathbb{R}^6$:
    $$ s = [x, \theta_1, \theta_2, \dot{x}, \dot{\theta}_1, \dot{\theta}_2]^T $$
    
    Equations of Motion:
    --------------------
    Derived using Lagrangian Mechanics ($L = T - V$).
    The system dynamics are given by:
    $$ M(q) \ddot{q} + C(q, \dot{q}) + G(q) = F_{ext} $$
    
    Where:
    - $q = [x, \theta_1, \theta_2]^T$ are the generalized coordinates.
    - $M(q)$ is the symmetric, positive-definite mass matrix.
    - $C(q, \dot{q})$ represents Coriolis and centrifugal forces.
    - $G(q)$ represents gravitational forces.
    - $F_{ext} = [F, 0, 0]^T$ is the external force applied to the cart.
    
    Modular Design:
    ---------------
    - **ControlStrategy**: Determines how actions $a_t$ map to force $F$.
    - **RewardStrategy**: Determines the reward signal $r_t$.
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
        self.m1 = 0.5     # Mass of pole 1
        self.m2 = 0.5     # Mass of pole 2
        self.l1 = 1.0     # Length of pole 1
        self.l2 = 1.0     # Length of pole 2
        self.g = 9.81     # Gravity
        
        self.dt = 0.005
        self.wind_std = wind_std
        self.reset_mode = reset_mode
        self.current_impulse = 0.0
        
        # Action Space (Delegated)
        self.action_space = self.control_strategy.get_action_space()
    
        # Observation Space: [x, sin(t1), sin(t2), cos(t1), cos(t2), x_dot, t1_dot, t2_dot]
        high = np.array([
            5.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.state: np.ndarray = np.zeros(6, dtype=np.float32)
        
        # Friction (Curriculum controlled via Env or Strategy? 
        # Strategy usually controls Reward Curriculum.
        # Physics Curriculum (Friction/Gravity) is ENV property.
        # So we keep set_curriculum here affecting physics, and also pass it to reward strategy.
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
        
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        
        mode = self.reset_mode
        if options and "mode" in options:
            mode = options["mode"]
            
        if mode == "up":
            self.state[1] += np.pi
            self.state[2] += np.pi
        elif mode == "down":
            pass 
        elif mode == "random":
            self.state[1] = self.np_random.uniform(0, 2*np.pi)
            self.state[2] = self.np_random.uniform(0, 2*np.pi)
            
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = self.state
        return np.array([
            x,
            np.sin(theta1), np.cos(theta1),
            np.sin(theta2), np.cos(theta2),
            x_dot, theta1_dot, theta2_dot
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Get Force from Strategy
        # Env params needed by strategy? (e.g. current velocity for P-control)
        # We pass the full state and let strategy parse it.
        # But VelocityControl needs to know which index is velocity.
        # We can pass env_params dict.
        env_params = {
            'dt': self.dt,
            'velocity_index': 3, # x_dot is index 3 in State
            'max_force': 5000.0 # Clip param
        }
        
        force = self.control_strategy.get_force(action, self.state, env_params)
        
        # Wind / Impulse
        if self.wind_std > 0:
            force += self.np_random.normal(0, self.wind_std)
        force += self.current_impulse
        self.current_impulse = 0.0
        
        # 2. Integrate Physics
        self.state = self._rk4_step(self.state, force, self.dt)
        
        x = self.state[0]
        
        # 3. Terminate Logic (Position Only)
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
        Computes accelerations $\ddot{q}$ for Double Pendulum.
        """
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
        
        M, m1, m2 = self.M, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        c12 = np.cos(theta1 - theta2)
        s12 = np.sin(theta1 - theta2)
        
        M_mat = np.array([
            [M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2],
            [(m1 + m2) * l1 * c1, (m1 + m2) * l1**2, m2 * l1 * l2 * c12],
            [m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2**2]
        ])
        
        damping_x = -self.friction_cart * x_dot
        damping_t1 = -self.friction_pole * theta1_dot
        damping_t2 = -self.friction_pole * theta2_dot
        
        C_vec = np.array([
            -(m1 + m2) * l1 * s1 * theta1_dot**2 - m2 * l2 * s2 * theta2_dot**2,
            m2 * l1 * l2 * s12 * theta2_dot**2,
            -m2 * l1 * l2 * s12 * theta1_dot**2
        ])
        
        G_vec = np.array([
            0,
            -(m1 + m2) * g * l1 * s1,
            -m2 * g * l2 * s2
        ])
        
        D_vec = np.array([damping_x, damping_t1, damping_t2])
        F_vec = np.array([force, 0, 0])
        
        RHS = F_vec + D_vec - C_vec - G_vec
        q_dd = np.linalg.solve(M_mat, RHS)
        
        return np.concatenate(([x_dot, theta1_dot, theta2_dot], q_dd))

    def _get_energy(self) -> float:
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = self.state
        M, m1, m2 = self.M, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        
        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        c12 = np.cos(theta1 - theta2)
        
        M11 = M + m1 + m2
        M12 = (m1 + m2) * l1 * c1
        M13 = m2 * l2 * c2
        M22 = (m1 + m2) * l1**2
        M23 = m2 * l1 * l2 * c12
        M33 = m2 * l2**2
        
        T = 0.5 * (
            M11 * x_dot**2 +
            M22 * theta1_dot**2 +
            M33 * theta2_dot**2 +
            2 * M12 * x_dot * theta1_dot +
            2 * M13 * x_dot * theta2_dot +
            2 * M23 * theta1_dot * theta2_dot
        )
        V = -(m1 + m2) * g * l1 * c1 - m2 * g * l2 * c2 # Potential is 0 at pivot?
                                                       # Original: m1 g l1 (1 - cos t1) + m2 g (l1(1-c1) + l2(1-c2))
                                                       # But often PE = -mgh 
                                                       # y1 = -l1 c1. y2 = -l1 c1 - l2 c2.
                                                       # V = m1 g y1 + m2 g y2
        V = m1 * g * (-l1 * c1) + m2 * g * (-l1 * c1 - l2 * c2)
        return T + V

    def render(self):
        pass

def angle_normalize(x, type="rad"):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
