# Double Pendulum Stabilization: Retrospective & Analysis

## Objective
Achieve robust stabilization of the Double Pendulum on a Cart in the "Up-Up" equilibrium position.

## Summary of Attempts

### Phase 1: Initial Setup & Physics
*   **Configuration**: Cart Mass $M=10.0$kg, Force $F=10.0$N.
*   **Result**: Agent lacked control authority. The heavy cart and weak force meant the agent could not recover from even small deviations.
*   **Adjustment**: Increased Force to $50.0$N.

### Phase 2: Reward Function Iterations
*   **Energy-Based**: Reward based on Total Energy $E \to E_{target}$.
    *   *Issue*: Agent learned to swing up but not stabilize. Energy matching doesn't enforce specific state (x=0, theta=0).
*   **Survival**: Reward $= +1$ for every step $|theta| < threshold$.
    *   *Issue*: "Suicide Policy". Agent would quickly move to a state where it falls slowly to maximize steps, rather than balancing.
*   **LQR-Inspired (Current)**: $R = R_{alive} - (w_a \theta^2 + w_v \dot{\theta}^2 + w_x x^2 + w_u u^2)$.
    *   *Result*: Better, but sensitive to weights.

### Phase 3: Advanced Tuning (The "Fighting" Agent)
*   **Physics Refinement**:
    *   Reduced Time Step $dt = 0.005s$ (200Hz) to capture high-frequency dynamics.
    *   **Mass Reduction**: Reduced Cart Mass $M=10.0 \to 1.0$kg.
        *   *Hypothesis*: Higher Force/Mass ratio ($50N/1kg = 50m/s^2$) gives "superhuman" control authority.
*   **RL Improvements**:
    *   **Action Scaling**: Fixed critical bug where PPO output $[-1, 1]$ was not scaled by `force_mag`.
    *   **Observation Space**: Switched to $[\sin(\theta), \cos(\theta), \dots]$ to avoid discontinuity at $\pi$.
    *   **Network**: Increased capacity to `[512, 512]`.
# Double Pendulum Stabilization: Retrospective & Analysis

## Objective
Achieve robust stabilization of the Double Pendulum on a Cart in the "Up-Up" equilibrium position.

## Summary of Attempts

### Phase 1: Initial Setup & Physics
*   **Configuration**: Cart Mass $M=10.0$kg, Force $F=10.0$N.
*   **Result**: Agent lacked control authority. The heavy cart and weak force meant the agent could not recover from even small deviations.
*   **Adjustment**: Increased Force to $50.0$N.

### Phase 2: Reward Function Iterations
*   **Energy-Based**: Reward based on Total Energy $E \to E_{target}$.
    *   *Issue*: Agent learned to swing up but not stabilize. Energy matching doesn't enforce specific state (x=0, theta=0).
*   **Survival**: Reward $= +1$ for every step $|theta| < threshold$.
    *   *Issue*: "Suicide Policy". Agent would quickly move to a state where it falls slowly to maximize steps, rather than balancing.
*   **LQR-Inspired (Current)**: $R = R_{alive} - (w_a \theta^2 + w_v \dot{\theta}^2 + w_x x^2 + w_u u^2)$.
    *   *Result*: Better, but sensitive to weights.

### Phase 3: Advanced Tuning (The "Fighting" Agent)
*   **Physics Refinement**:
    *   Reduced Time Step $dt = 0.005s$ (200Hz) to capture high-frequency dynamics.
    *   **Mass Reduction**: Reduced Cart Mass $M=10.0 \to 1.0$kg.
        *   *Hypothesis*: Higher Force/Mass ratio ($50N/1kg = 50m/s^2$) gives "superhuman" control authority.
*   **RL Improvements**:
    *   **Action Scaling**: Fixed critical bug where PPO output $[-1, 1]$ was not scaled by `force_mag`.
    *   **Observation Space**: Switched to $[\sin(\theta), \cos(\theta), \dots]$ to avoid discontinuity at $\pi$.
    *   **Network**: Increased capacity to `[512, 512]`.
    *   **Exploration**: Implemented Ornstein-Uhlenbeck (OU) Noise for temporally correlated exploration.

## 5. The Solution: Ratchet Curriculum & Exponential Reward
Following these failures, we pivoted to a **Curriculum Learning** approach ("The Ratchet") combined with an **Exponential Continuity Reward**.

*   **Why it works**:
    *   **Curriculum**: Prevents the agent from being overwhelmed by chaos early on.
    *   **Ratchet**: Ensures mastery before increasing difficulty.
    *   **Exponential Reward**: Provides a clear, strong signal for stabilization that outweighs the noise.

See [docs/stabilization_strategy.md](stabilization_strategy.md) for the full solution details.

## Phase 6: Visualization Refactoring & Physics Update (Double Length)
*   **Physics Update**:
    *   **Pendulum Length**: Doubled from $l=0.5$m to $l=1.0$m.
    *   **Reasoning**: Longer pendulums have a lower natural frequency ($\omega \propto \sqrt{g/l}$), making them slower and theoretically easier to balance (more reaction time). However, they require more energy to swing up ($E_{pot} \propto l$).
*   **Visualization Pipeline**:
    *   **Centralized Script**: `src/generate_report.py` orchestrates the entire process.
    *   **Random Seeds**: Removed magic numbers. The script picks a random seed and passes it to all subprocesses (`visualize_overlay.py`, `simulate.py`), ensuring the "Overlay" and "Final Run" videos are mathematically identical.
    *   **Side-by-Side Verification**: Automatically generates `comparison.mp4` to prove consistency.
*   **Documentation**:
    *   Updated `README.md` and `docs/` to reflect the new physics and pipeline.

## References
1.  [OpenAI Gym Documentation](https://www.gymlibrary.dev/)
2.  [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
3.  [Underactuated Robotics (Tedrake)](http://underactuated.mit.edu/)
4.  [Double Pendulum Chaos (YouTube)](https://www.youtube.com/watch?v=pWekXMZJ2zM)
