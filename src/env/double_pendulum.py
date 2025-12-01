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

    def __init__(self, render_mode: Optional[str] = None, wind_std: float = 0.0, reset_mode: str = "up"):
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
        self.wind_std = wind_std # Standard deviation of wind force noise
        self.reset_mode = reset_mode # "up" or "down"
        self.current_impulse = 0.0 # Instantaneous impulse force
        
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
        
        if self.reset_mode == "up":
            self.state[1] += np.pi # theta1 near pi
            self.state[2] += np.pi # theta2 near pi
        elif self.reset_mode == "down":
            # Start near 0 (down)
            pass
        else:
            raise ValueError(f"Unknown reset_mode: {self.reset_mode}")
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        
        # Add Wind (Continuous Noise)
        if self.wind_std > 0:
            force += self.np_random.normal(0, self.wind_std)
            
        # Add Impulse (Instantaneous)
        force += self.current_impulse
        self.current_impulse = 0.0 # Reset impulse after one step
        
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
        
        # Reward Function
        # Goal: Stabilize at theta1 = pi, theta2 = pi
        # Normalize angles to [-pi, pi]
        t1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
        t2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
        
        # Errors (target is pi, which maps to 0 in this shifted space if we shifted correctly? 
        # No, let's just use cos distance for robustness against wrapping)
        # Target: Upright (pi). cos(pi) = -1.
        # We want to maximize: -(theta - pi)^2 approx (cos(theta) - (-1)) = cos(theta) + 1 ?
        # No, if theta=pi, cos(pi)=-1. We want to minimize distance to -1.
        # Let's use: R = -(theta_err)^2.
        
        # Angle error from pi
        t1_err = angle_normalize(theta1 - np.pi)
        t2_err = angle_normalize(theta2 - np.pi)
        
        # Weights
        # Weights
        # w3 (position) increased to 0.5 to encourage re-centering
        w1, w2, w3, w4 = 1.0, 1.0, 0.5, 0.01
        
        # Gaussian Reward (Positive)
        # R = exp(-distance^2)
        
        # 1. Spatial Reward (Alignment)
        dist_sq = (
            w1 * t1_err**2 + 
            w2 * t2_err**2 + 
            w3 * x**2 + 
            w4 * (theta1_dot**2 + theta2_dot**2)
        )
        r_spatial = np.exp(-dist_sq)
        
        # 2. Energy Reward (Shaping)
        # Target Energy (Up-Up Equilibrium: theta1=pi, theta2=pi, velocities=0)
        # V_target = -(m1+m2)g l1 cos(pi) - m2 g l2 cos(pi) = (m1+m2)g l1 + m2 g l2
        target_energy = (self.m1 + self.m2) * self.g * self.l1 + self.m2 * self.g * self.l2
        current_energy = self._get_energy()
        
        # Energy difference
        energy_diff = np.abs(current_energy - target_energy)
        
        # Scale energy difference? Energy can be large (~40J).
        # We want the kernel width to be reasonable.
        # If diff is 10J, exp(-10) is tiny.
        # Let's normalize or scale sigma.
        # sigma_e = 10.0
        r_energy = np.exp(-energy_diff / 10.0)
        
        # Combined Reward
        # Hybrid: w_spatial * r_spatial + w_energy * r_energy
        w_spatial = 0.6
        w_energy = 0.4
        
        reward = w_spatial * r_spatial + w_energy * r_energy

 
        
        return self._get_obs(), reward, terminated, truncated, {}

    def apply_impulse(self, force: float):
        """Apply an instantaneous force to the cart."""
        self.current_impulse = force

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

    def _get_energy(self) -> float:
        """Computes Total Energy (T + V) of the system."""
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = self.state
        
        # Constants
        M, m1, m2 = self.M, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        
        # Kinetic Energy T = 0.5 * q_dot.T * M(q) * q_dot
        # We can reuse the M_mat calculation or simplify.
        # Let's reuse the logic from _dynamics for consistency, but optimized.
        
        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        c12 = np.cos(theta1 - theta2)
        
        # Mass Matrix M(q)
        M11 = M + m1 + m2
        M12 = (m1 + m2) * l1 * c1
        M13 = m2 * l2 * c2
        M22 = (m1 + m2) * l1**2
        M23 = m2 * l1 * l2 * c12
        M33 = m2 * l2**2
        
        # T = 0.5 * (M11 xd^2 + M22 th1d^2 + M33 th2d^2 + 2 M12 xd th1d + 2 M13 xd th2d + 2 M23 th1d th2d)
        T = 0.5 * (
            M11 * x_dot**2 +
            M22 * theta1_dot**2 +
            M33 * theta2_dot**2 +
            2 * M12 * x_dot * theta1_dot +
            2 * M13 * x_dot * theta2_dot +
            2 * M23 * theta1_dot * theta2_dot
        )
        
        # Potential Energy V
        # 0 is Down (theta=0). Up is theta=pi.
        # V = -m g h.
        # h1 = -l1 cos(theta1)
        # h2 = -l1 cos(theta1) - l2 cos(theta2)
        # V = m1 g h1 + m2 g h2
        # V = -(m1+m2) g l1 cos(theta1) - m2 g l2 cos(theta2)
        
        V = -(m1 + m2) * g * l1 * c1 - m2 * g * l2 * c2
        
        return T + V

    def render(self):
        pass

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
