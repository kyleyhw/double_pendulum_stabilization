import numpy as np
import scipy.linalg

class LQRController:
    def __init__(self, env):
        self.env = env
        self.g = env.g
        self.m1 = env.m1
        self.m2 = env.m2
        self.M = env.M
        self.l1 = env.l1
        self.l2 = env.l2
        
        # Linearize around the UP-UP equilibrium point
        # State: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        # Equilibrium: [0, pi, pi, 0, 0, 0]
        # Note: In our env, 0 is DOWN, pi is UP.
        
        self.K = self._compute_lqr_gain()
        
    def _compute_lqr_gain(self):
        # System parameters
        M = self.M
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        g = self.g
        
        # Linearized Dynamics Matrix A (dx/dt = Ax + Bu)
        # Derived from EOM at vertical equilibrium
        # This is complex for double pendulum, using standard results or numerical linearization
        # Let's use numerical linearization for robustness
        
        A = np.zeros((6, 6))
        B = np.zeros((6, 1))
        
        # Small perturbation
        epsilon = 1e-4
        
        # Equilibrium state (Upright)
        x_eq = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
        u_eq = np.array([0.0])
        
        # Compute A = df/dx
        for i in range(6):
            x_plus = x_eq.copy()
            x_plus[i] += epsilon
            x_minus = x_eq.copy()
            x_minus[i] -= epsilon
            
            f_plus = self._dynamics(x_plus, u_eq)
            f_minus = self._dynamics(x_minus, u_eq)
            
            A[:, i] = (f_plus - f_minus) / (2 * epsilon)
            
        # Compute B = df/du
        u_plus = u_eq.copy()
        u_plus[0] += epsilon
        u_minus = u_eq.copy()
        u_minus[0] -= epsilon
        
        f_plus = self._dynamics(x_eq, u_plus)
        f_minus = self._dynamics(x_eq, u_minus)
        
        B[:, 0] = (f_plus - f_minus) / (2 * epsilon)
        
        print("Linearized A matrix:\n", A)
        print("Linearized B matrix:\n", B)
        
        # LQR Weights
        Q = np.diag([10.0, 100.0, 100.0, 1.0, 1.0, 1.0]) # High penalty on angles
        R = np.array([[0.1]]) # Low penalty on control effort
        
        # Solve Riccati Equation
        try:
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            # Compute Gain K
            K = np.linalg.inv(R) @ B.T @ P
            return K
        except Exception as e:
            print(f"LQR Solver failed: {e}")
            # Return zero gain as fallback to prevent crash
            return np.zeros((1, 6))
    
    def _dynamics(self, state, action):
        # Helper to get x_dot from env dynamics
        # We need to bypass the env.step() integration and get instantaneous x_dot
        # State: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
        
        # Unpack
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
        F = action[0]
        
        # Mass Matrix M(q)
        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        c12 = np.cos(theta1 - theta2)
        s1 = np.sin(theta1)
        s2 = np.sin(theta2)
        s12 = np.sin(theta1 - theta2)
        
        m1, m2, M, l1, l2, g = self.m1, self.m2, self.M, self.l1, self.l2, self.g
        
        M_mat = np.array([
            [M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2],
            [(m1 + m2) * l1 * c1, (m1 + m2) * l1**2, m2 * l1 * l2 * c12],
            [m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2**2]
        ])
        
        # Coriolis & Gravity
        C_vec = np.array([
            -(m1 + m2) * l1 * s1 * theta1_dot**2 - m2 * l2 * s2 * theta2_dot**2,
            m2 * l1 * l2 * s12 * theta2_dot**2 - (m1 + m2) * g * l1 * s1,
            -m2 * l1 * l2 * s12 * theta1_dot**2 - m2 * g * l2 * s2
        ])
        
        # Force Vector
        tau = np.array([F, 0, 0])
        
        # Solve for accelerations: M * q_ddot + C = tau  =>  q_ddot = M_inv * (tau - C)
        # Note: My C_vec includes Gravity terms but with signs adjusted for the equation M*q_ddot + ... = tau
        # Let's be careful with signs.
        # From derivation: M q_ddot + C_coriolis + G = tau
        # So q_ddot = M_inv * (tau - C_coriolis - G)
        
        # Re-calculating terms exactly as in env
        # C(q, q_dot)
        coriolis = np.array([
            -(m1 + m2) * l1 * s1 * theta1_dot**2 - m2 * l2 * s2 * theta2_dot**2,
            m2 * l1 * l2 * s12 * theta2_dot**2,
            -m2 * l1 * l2 * s12 * theta1_dot**2
        ])
        
        gravity = np.array([
            0,
            -(m1 + m2) * g * l1 * s1, # Note: Potential V = -mgl cos(theta). G = dV/dtheta = mgl sin(theta).
            -m2 * g * l2 * s2         # But in env we used +y up? Let's check env implementation.
        ])
        
        # In Env:
        # b = np.array([
        #     F + (self.m1 + self.m2) * self.l1 * s1 * theta1_dot**2 + self.m2 * self.l2 * s2 * theta2_dot**2,
        #     -self.m2 * self.l1 * self.l2 * s12 * theta2_dot**2 + (self.m1 + self.m2) * self.g * self.l1 * s1,
        #     self.m2 * self.l1 * self.l2 * s12 * theta1_dot**2 + self.m2 * self.g * self.l2 * s2
        # ])
        # This matches q_ddot = M_inv * b
        
        # Add small damping to linearization to help LQR solver
        # This moves eigenvalues off the imaginary axis
        damping = 0.1
        
        b = np.array([
            F + (m1 + m2) * l1 * s1 * theta1_dot**2 + m2 * l2 * s2 * theta2_dot**2 - damping * x_dot,
            -m2 * l1 * l2 * s12 * theta2_dot**2 - (m1 + m2) * g * l1 * s1 - damping * theta1_dot,
            m2 * l1 * l2 * s12 * theta1_dot**2 - m2 * g * l2 * s2 - damping * theta2_dot
        ])
        
        q_ddot = np.linalg.solve(M_mat, b)
        
        return np.concatenate([state[3:], q_ddot])

    def get_action(self, state):
        # u = -K (x - x_eq)
        x_eq = np.array([0, np.pi, np.pi, 0, 0, 0])
        
        # Normalize angles to be close to pi
        # If theta is -pi, it's same as pi.
        # We need the error state delta_x
        delta_x = state - x_eq
        
        # Wrap angles to [-pi, pi]
        delta_x[1] = (delta_x[1] + np.pi) % (2 * np.pi) - np.pi
        delta_x[2] = (delta_x[2] + np.pi) % (2 * np.pi) - np.pi
        
        u = -self.K @ delta_x
        return u
