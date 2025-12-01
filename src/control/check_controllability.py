import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.env.double_pendulum import DoublePendulumCartEnv

def check_controllability():
    env = DoublePendulumCartEnv()
    
    # Define Equilibria (x, theta1, theta2, dx, dtheta1, dtheta2)
    # Note: Angles in env are continuous, but for linearization we use 0/pi.
    equilibria = {
        "Down-Down (Stable)":   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "Up-Up (Unstable)":     np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]),
        "Down-Up (Unstable)":   np.array([0.0, 0.0, np.pi, 0.0, 0.0, 0.0]),
        "Up-Down (Unstable)":   np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
    }
    
    print("# Controllability Analysis\n")
    print(f"System Dimension (n): 6")
    print(f"Control Dimension (m): 1")
    print("-" * 60)
    
    results = {}
    
    for name, state in equilibria.items():
        print(f"\nAnalyzing: {name}")
        
        # 1. Linearize (Numerical Jacobian)
        # x_dot = f(x, u)
        # A = df/dx, B = df/du
        
        eps = 1e-5
        A = np.zeros((6, 6))
        B = np.zeros((6, 1))
        
        # Compute A
        for i in range(6):
            x_plus = state.copy()
            x_plus[i] += eps
            x_minus = state.copy()
            x_minus[i] -= eps
            
            f_plus = env._dynamics(x_plus, 0.0)
            f_minus = env._dynamics(x_minus, 0.0)
            
            A[:, i] = (f_plus - f_minus) / (2 * eps)
            
        # Compute B
        u_plus = eps
        u_minus = -eps
        f_plus = env._dynamics(state, u_plus)
        f_minus = env._dynamics(state, u_minus)
        B[:, 0] = (f_plus - f_minus) / (2 * eps)
        
        # 2. Compute Controllability Matrix
        # C = [B, AB, A^2B, ..., A^(n-1)B]
        n = 6
        C_list = [B]
        curr_term = B
        for _ in range(1, n):
            curr_term = A @ curr_term
            C_list.append(curr_term)
            
        C = np.hstack(C_list)
        
        # 3. Check Rank
        rank = np.linalg.matrix_rank(C)
        
        # 4. Check Eigenvalues (Stability)
        eigvals = np.linalg.eigvals(A)
        unstable_modes = np.sum(np.real(eigvals) > 1e-5)
        
        is_controllable = (rank == n)
        
        print(f"  Rank of C: {rank} / 6")
        print(f"  Controllable: {'YES' if is_controllable else 'NO'}")
        print(f"  Unstable Modes (Re > 0): {unstable_modes}")
        print(f"  Eigenvalues: {np.round(eigvals, 2)}")
        
        results[name] = {
            "controllable": is_controllable,
            "rank": rank,
            "eigenvalues": eigvals
        }

    return results

if __name__ == "__main__":
    check_controllability()
