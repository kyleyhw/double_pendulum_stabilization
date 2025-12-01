import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

class DoublePendulumCartEnv(gym.Env):
    """
    Double Pendulum on a Cart Environment.
    
    System:
        - Cart (M) moving on x-axis.
        - Pole 1 (m1, l1) attached to cart.
        - Pole 2 (m2, l2) attached to Pole 1.
        
    State:
        [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        
    Action:
        Force F applied to the cart.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # System Parameters
        self.M = 1.0      # Mass of cart [kg]
        self.m1 = 0.5     # Mass of pole 1 [kg]
        self.m2 = 0.5     # Mass of pole 2 [kg]
        self.l1 = 1.0     # Length of pole 1 [m]
        self.l2 = 1.0     # Length of pole 2 [m]
        self.g = 9.81     # Gravity [m/s^2]
        
        self.dt = 0.02    # Time step [s]
        self.force_mag = 20.0 # Max force magnitude
        
        # Action Space: Continuous force [-F_max, F_max]
        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )
        
        # Observation Space: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        # Limits: x is limited, angles are unbounded (trig used in reward), velocities limited
        high = np.array([
            5.0,                # x limit
            np.inf,             # theta1
            np.inf,             # theta2
            np.inf,             # x_dot
            np.inf,             # theta1_dot
            np.inf              # theta2_dot
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.state = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize state: slightly perturbed from vertical up (theta=pi) or down (theta=0)?
        # Goal is to stabilize UP. 
        # Let's define UP as theta = pi (180 deg) based on our derivation where 0 is DOWN.
        # Or we can initialize near DOWN and swing up (harder).
        # For "stabilization" task, we usually start near the equilibrium.
        
        # Initial state: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        # Start near Upright: theta1 ~ pi, theta2 ~ pi
        
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.state[1] += np.pi # theta1 near pi
        self.state[2] += np.pi # theta2 near pi
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        
        # RK4 Integration
        self.state = self._rk4_step(self.state, force, self.dt)
        
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = self.state
        
        # Normalize angles to [-pi, pi] for reward calculation if needed, 
        # but state keeps continuous angle.
        
        # Termination conditions
        terminated = bool(
            x < -self.observation_space.high[0]
            or x > self.observation_space.high[0]
        )
        truncated = False
        
        # Reward Function (Placeholder for Phase 2)
        # Penalize distance from upright (pi) and cart center (0)
        # theta_err = angle_normalize(theta - pi)
        reward = 0.0 
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _rk4_step(self, state: np.ndarray, force: float, dt: float) -> np.ndarray:
        """Runge-Kutta 4th Order Integration"""
        k1 = self._dynamics(state, force)
        k2 = self._dynamics(state + 0.5 * dt * k1, force)
        k3 = self._dynamics(state + 0.5 * dt * k2, force)
        k4 = self._dynamics(state + dt * k3, force)
        
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state

    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        """
        Computes y_dot = f(y, u)
        y = [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        """
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
        
        # Unpack constants
        M, m1, m2 = self.M, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        
        # Precompute trig terms
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        c12 = np.cos(theta1 - theta2)
        s12 = np.sin(theta1 - theta2)
        
        # Mass Matrix M(q)
        # [ M+m1+m2,    (m1+m2)l1 c1,    m2 l2 c2 ]
        # [ (m1+m2)l1 c1, (m1+m2)l1^2,   m2 l1 l2 c12 ]
        # [ m2 l2 c2,     m2 l1 l2 c12,  m2 l2^2 ]
        
        M_mat = np.array([
            [M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2],
            [(m1 + m2) * l1 * c1, (m1 + m2) * l1**2, m2 * l1 * l2 * c12],
            [m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2**2]
        ])
        
        # Coriolis & Gravity Vector B(q, q_dot)
        # C terms:
        # 1: -(m1+m2)l1 s1 theta1_dot^2 - m2 l2 s2 theta2_dot^2
        # 2: m2 l1 l2 s12 theta2_dot^2
        # 3: -m2 l1 l2 s12 theta1_dot^2
        
        # G terms:
        # 1: 0
        # 2: (m1+m2)g l1 s1
        # 3: m2 g l2 s2
        
        # Note: In derivation, G was on LHS. So on RHS it is -G.
        # But wait, derivation said M q_dd + C + G = F.
        # So M q_dd = F - C - G.
        
        C_vec = np.array([
            -(m1 + m2) * l1 * s1 * theta1_dot**2 - m2 * l2 * s2 * theta2_dot**2,
            m2 * l1 * l2 * s12 * theta2_dot**2,
            -m2 * l1 * l2 * s12 * theta1_dot**2
        ])
        
        G_vec = np.array([
            0,
            (m1 + m2) * g * l1 * s1,
            m2 * g * l2 * s2
        ])
        # Wait, check signs of G.
        # V = -(m1+m2)g l1 cos(theta1) ...
        # dV/dtheta1 = (m1+m2)g l1 sin(theta1).
        # This is the G term.
        # So M q_dd + ... + G = F.
        # So q_dd = M_inv * (F_vec - C_vec - G_vec).
        
        F_vec = np.array([force, 0, 0])
        
        # Solve for q_dd
        RHS = F_vec - C_vec - G_vec
        q_dd = np.linalg.solve(M_mat, RHS)
        
        return np.concatenate(([x_dot, theta1_dot, theta2_dot], q_dd))

    def render(self):
        pass
