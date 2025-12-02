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
        self.M = 1.0      # Mass of cart [kg] (Reduced for authority)
        self.m1 = 0.5     # Mass of pole 1 [kg]
        self.m2 = 0.5     # Mass of pole 2 [kg]
        self.l1 = 1.0     # Length of pole 1 [m] (Doubled)
        self.l2 = 1.0     # Length of pole 2 [m] (Doubled)
        self.g = 9.81     # Gravity [m/s^2]
        
        self.dt = 0.005   # Time step [s] (200Hz for stability)
        self.force_mag = 50.0 # Max force magnitude (Increased for authority)
        self.wind_std = wind_std # Standard deviation of wind force noise
        self.reset_mode = reset_mode # "up" or "down"
        self.current_impulse = 0.0 # Instantaneous impulse force
        
        # Action Space: Continuous force [-F_max, F_max]
        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )
        
        # Observation Space: [x, sin(t1), cos(t1), sin(t2), cos(t2), x_dot, t1_dot, t2_dot]
        # Range: x=[-5,5], trig=[-1,1], vels=[-inf, inf]
        high = np.array([
            5.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.state = None
        
        # Curriculum Learning Parameters
        self.curriculum_alpha = 0.0
        self.sigma_target = 1.0
        self.sigma_start = 5.0
        
        # Potential-Based Shaping
        self.last_potential = None
        self.gamma = 0.99 # Discount factor for shaping
        
        # Initialize Curriculum Params
        self.friction_cart = 0.0
        self.friction_pole = 0.0
        self.reward_threshold = 1.5708
        
        # Continuity Counter
        self.steps_above_threshold = 0

    def set_curriculum(self, difficulty: float):
        difficulty = np.clip(difficulty, 0.0, 1.0)
        
        # Gravity: 2.0 -> 9.81
        self.g = 2.0 + difficulty * (9.81 - 2.0)
        
        # Friction: 0.5 -> 0.0 (Cart), 0.1 -> 0.0 (Pole)
        self.friction_cart = 0.5 * (1.0 - difficulty)
        self.friction_pole = 0.1 * (1.0 - difficulty)
        
        # Reward Threshold: 90 deg (1.57 rad) -> 10 deg (0.17 rad)
        start_angle = np.pi / 2
        end_angle = np.deg2rad(10)
        self.reward_threshold = start_angle - difficulty * (start_angle - end_angle)
        
        return {
            "g": self.g,
            "friction_cart": self.friction_cart,
            "reward_threshold_deg": np.rad2deg(self.reward_threshold)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize state
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        
        mode = self.reset_mode
        if options and "mode" in options:
            mode = options["mode"]
            
        if mode == "up":
            self.state[1] += np.pi
            self.state[2] += np.pi
        elif mode == "down":
            pass # Already near 0
        elif mode == "random":
            self.state[1] = self.np_random.uniform(0, 2*np.pi)
            self.state[2] = self.np_random.uniform(0, 2*np.pi)
            
        self.last_potential = 0.0
        self.steps_above_threshold = 0
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
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        
        if self.wind_std > 0:
            force += self.np_random.normal(0, self.wind_std)
            
        force += self.current_impulse
        self.current_impulse = 0.0
        
        self.state = self._rk4_step(self.state, force, self.dt)
        
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = self.state
        
        terminated = bool(
            x < -self.observation_space.high[0]
            or x > self.observation_space.high[0]
        )
        
        # Terminate if either pole falls below horizontal (Phase 5: Time Above Horizontal)
        # Target is PI. Horizontal is PI +/- PI/2.
        # So we want |theta - PI| < PI/2
        # BUT user requested: "stop killing the simulations prematurely unless it hits the edges"
        # So we REMOVE the angle termination!
        
        truncated = False
        
        # --- Reward Calculation (Phase 5: Time Above Horizontal + Exponential Continuity) ---
        # User Request: "exponentiate to blow up the continuous period rewarding even more"
        
        t1_err = angle_normalize(theta1 - np.pi)
        t2_err = angle_normalize(theta2 - np.pi)
        
        if abs(t1_err) < self.reward_threshold and abs(t2_err) < self.reward_threshold:
            self.steps_above_threshold += 1
            
            # Exponential Reward: exp(time_above) - 1.0
            time_above = self.steps_above_threshold * self.dt
            reward = np.exp(time_above) - 1.0
            
            # Gaussian Position Penalty (Prevent Suiciding)
            # Center at 0, sigma=1.0 (5 std devs to edge at 5.0)
            sigma_x = 1.0
            pos_penalty = np.exp(-(x**2) / (2 * sigma_x**2))
            reward *= pos_penalty
        else:
            self.steps_above_threshold = 0
            reward = 0.0
        
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
        M_mat = np.array([
            [M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2],
            [(m1 + m2) * l1 * c1, (m1 + m2) * l1**2, m2 * l1 * l2 * c12],
            [m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2**2]
        ])
        
        # Damping Terms (Curriculum)
        damping_x = -self.friction_cart * x_dot
        damping_t1 = -self.friction_pole * theta1_dot
        damping_t2 = -self.friction_pole * theta2_dot
        
        # Coriolis & Gravity Vector C(q, q_dot) + G(q)
        # Note: In derivation, M q_dd + C + G = F
        # So q_dd = M_inv * (F - C - G)
        
        C_vec = np.array([
            -(m1 + m2) * l1 * s1 * theta1_dot**2 - m2 * l2 * s2 * theta2_dot**2 - damping_x,
            m2 * l1 * l2 * s12 * theta2_dot**2 - (m1 + m2) * g * l1 * s1 - damping_t1,
            -m2 * l1 * l2 * s12 * theta1_dot**2 - m2 * g * l2 * s2 - damping_t2
        ])
        
        # Wait, G terms signs?
        # V = -(m1+m2)g l1 c1 - m2 g l2 c2
        # dV/dt1 = (m1+m2)g l1 s1. This is G1.
        # dV/dt2 = m2 g l2 s2. This is G2.
        # So G vector is positive.
        # So F - C - G is correct if C and G are on LHS.
        # My C_vec implementation above includes G terms?
        # Let's check previous implementation.
        # Previous implementation had:
        # C_vec = [ ... ]
        # G_vec = [ 0, (m1+m2)g l1 s1, m2 g l2 s2 ]
        # RHS = F - C - G.
        # Here I combined them into C_vec?
        # C_vec[1] = ... - (m1+m2)g l1 s1.
        # If I subtract this C_vec from F, I get F - (... - G) = F - ... + G.
        # That would be WRONG.
        # I need to subtract G.
        # So if I include G in C_vec, it should be +G.
        # Then F - C_vec = F - (... + G) = F - ... - G. Correct.
        # So C_vec[1] should have + (m1+m2)g l1 s1.
        # BUT wait, the previous code had:
        # C_vec = [ ... ]
        # G_vec = [ ... ]
        # RHS = F - C - G.
        # So G terms are positive in G_vec.
        # So if I put them in C_vec, they should be positive.
        # My code above has `- (m1+m2) * g * l1 * s1`.
        # This means F - C_vec = F - (... - G) = F - ... + G.
        # This effectively puts gravity on the RHS with a positive sign, which means gravity *helps* motion?
        # Gravity pulls down.
        # If theta=pi/2 (horizontal), sin=1. G term is positive.
        # Torque = -G.
        # So we need negative torque.
        # If I have +G on RHS, it accelerates theta positively (downwards).
        # Theta=0 is down. Theta=pi is up.
        # If theta=pi/2 (3 o'clock), gravity pulls to 0. So theta decreases.
        # So acceleration should be negative.
        # So we need -G on RHS.
        # So F - ... - G.
        # So G term should be positive in C_vec (if subtracting C_vec).
        # So `+ (m1+m2) * g * l1 * s1`.
        
        # Let's separate them to be safe and clear.
        
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
        
        D_vec = np.array([
            damping_x,
            damping_t1,
            damping_t2
        ])
        
        F_vec = np.array([force, 0, 0])
        
        # Solve for q_dd
        # M q_dd + C + G = F + D
        # M q_dd = F + D - C - G
        RHS = F_vec + D_vec - C_vec - G_vec
        q_dd = np.linalg.solve(M_mat, RHS)
        
        return np.concatenate(([x_dot, theta1_dot, theta2_dot], q_dd))

    def _get_energy(self) -> float:
        """Computes Total Energy (T + V) of the system."""
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = self.state
        
        # Constants
        M, m1, m2 = self.M, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        
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
        
        # T
        T = 0.5 * (
            M11 * x_dot**2 +
            M22 * theta1_dot**2 +
            M33 * theta2_dot**2 +
            2 * M12 * x_dot * theta1_dot +
            2 * M13 * x_dot * theta2_dot +
            2 * M23 * theta1_dot * theta2_dot
        )
        
        # V
        V = -(m1 + m2) * g * l1 * c1 - m2 * g * l2 * c2
        
        return T + V

    def render(self):
        pass

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
