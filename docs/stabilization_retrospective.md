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

## Current Status (Failure Analysis)
Despite high control authority and advanced RL tuning, the agent fails to achieve true stability.
*   **Symptom**: "Clearly falling". The agent survives for 3-5 seconds but exhibits "flailing" or "fighting" behavior rather than converging to a quiet equilibrium.
*   **Root Causes (Hypotheses)**:
    1.  **Reward Landscape**: The "Survival with Penalty" reward might still encourage "living on the edge" rather than perfect stability.
    2.  **Local Minima**: PPO might be stuck in a suboptimal policy that uses high-frequency actuation to fight gravity rather than finding the passive balance point.
    3.  **Physics/Sim Mismatch**: The aggressive control ($50m/s^2$) might be inducing numerical instability or unrealistic physics artifacts, even at 200Hz.

## Next Steps (The "Drawing Board")
1.  **Re-evaluate Physics**: Is $M=1.0$kg too light? Is $F=50$N too strong?
2.  **Control Theory Approach**: Implement a classic LQR controller first to prove stabilization is *possible* with current physics. If LQR can't do it, RL won't either.
3.  **Curriculum**: Start with a single pendulum, then double.
