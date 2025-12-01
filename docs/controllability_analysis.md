# Controllability Analysis of the Double Pendulum on a Cart

## Objective
To mathematically verify that the unstable equilibria (Up-Up, Down-Up, Up-Down) are controllable using the single actuator (cart force). This justifies the feasibility of the project.

## Methodology
We analyze the local controllability of the nonlinear system $\dot{x} = f(x, u)$ by linearizing it around each equilibrium point $x_e$.

1.  **Linearization**:
    $$ \dot{\delta x} = A \delta x + B \delta u $$
    Where $A = \frac{\partial f}{\partial x}|_{x_e}$ and $B = \frac{\partial f}{\partial u}|_{x_e}$.

2.  **Controllability Matrix**:
    $$ \mathcal{C} = [B, AB, A^2B, A^3B, A^4B, A^5B] $$

3.  **Kalman Rank Condition**:
    The system is locally controllable if and only if $\text{rank}(\mathcal{C}) = n$, where $n=6$ is the state dimension.

## Results

We computed the Jacobian matrices numerically and evaluated the rank of $\mathcal{C}$ for all four key configurations.

| Equilibrium | Configuration | Stability | Rank($\mathcal{C}$) | Controllable? | Eigenvalues (Stability) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Down-Down** | $\theta_1=0, \theta_2=0$ | Stable | **6 / 6** | **YES** | All Re($\lambda$) = 0 (Marginally Stable) |
| **Up-Up** | $\theta_1=\pi, \theta_2=\pi$ | Unstable | **6 / 6** | **YES** | 2 Unstable Modes (Re > 0) |
| **Down-Up** | $\theta_1=0, \theta_2=\pi$ | Unstable | **6 / 6** | **YES** | 1 Unstable Mode (Re > 0) |
| **Up-Down** | $\theta_1=\pi, \theta_2=0$ | Unstable | **6 / 6** | **YES** | 1 Unstable Mode (Re > 0) |

## Interpretation

### 1. Full Controllability
The fact that $\text{rank}(\mathcal{C}) = 6$ for all equilibria proves that **a solution exists**.
It is physically possible to steer the system from any point in the neighborhood of these equilibria to the equilibrium itself using only the cart.

### 2. Underactuation is not a Blocker
Although the system is underactuated (1 input, 3 degrees of freedom), the strong dynamic coupling (captured by the $AB, A^2B...$ terms) allows the single input to influence all state variables.
*   $B$ affects $x$ directly.
*   $AB$ affects $\theta_1$ (via cart acceleration).
*   $A^2B$ affects $\theta_2$ (via pole 1 coupling).

### 3. Conclusion
The project is **mathematically sound**. The unstable equilibria are stabilizable. The challenge lies purely in finding the nonlinear control policy (RL) that can navigate the global state space to reach these local basins of attraction.

## References
*   *Spong, M. W. (1995). The Swing Up Control Problem for the Acrobot.*
*   *Tedrake, R. (2023). Underactuated Robotics.*
