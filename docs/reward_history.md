# Reward Function Evolution

This document chronicles the iterative development of the reward function used to solve the Double Pendulum Stabilization task. Each iteration addressed specific failure modes observed during training.

## Iteration 1: The Naive Geometric Reward
**Goal**: Tell the agent "Up is Good".

$$ R = -(\theta_1 - \pi)^2 - (\theta_2 - \pi)^2 - 0.1 x^2 $$

*   **Logic**: Simple squared error penalty on angles and cart position.
*   **Result**: **Failure**.
    *   The agent learned to keep the cart centered but failed to swing up.
    *   The "gradient" of this reward is too shallow when the poles are hanging down. The agent didn't know *how* to get up.

## Iteration 2: The Energy Shaping Reward
**Goal**: Teach the agent to "Pump Energy".

$$ R_{energy} = \exp(-|E_{current} - E_{target}|) $$

*   **Logic**: The upright equilibrium has a specific total energy $E_{target}$. We reward the agent for having this energy, regardless of its position.
*   **Result**: **Partial Success**.
    *   The agent learned to swing up vigorously!
    *   **Problem**: The agent would reach the top but fly right past it. It learned to "orbit" the equilibrium, maintaining constant energy but never stopping.

## Iteration 3: Kinetic Damping
**Goal**: Teach the agent to "Stop at the Top".

$$ R_{kinetic} = \exp(-T) $$

*   **Logic**: Penalize Kinetic Energy ($T$) to encourage stillness.
*   **Result**: **Failure**.
    *   The agent learned to simply stop the pendulum at the bottom (Down-Down).
    *   Minimizing $T$ everywhere discourages the high-velocity swing-up needed to reach the top.

## Iteration 4: Gated Kinetic Damping
**Goal**: "Swing fast when down, stop when up."

$$ R_{kinetic} = \exp(-T) \cdot \underbrace{\exp(-(\theta - \pi)^2)}_{\text{Gate}} $$

*   **Logic**: Multiply the Kinetic Reward by a "Spatial Gate". The agent only gets paid for being still *if* it is also near the top.
*   **Result**: **Success (Unstable)**.
    *   The agent could stabilize, but often fell into "limit cycles" where it flailed around the top.
    *   The transition from "Swing" to "Stabilize" was too harsh.

## Iteration 5: Hybrid Reward with Curriculum (Current)
**Goal**: Smooth the learning curve.

$$ R_{total} = w_s R_{spatial}(\sigma) + w_e R_{energy}(\sigma) + w_k R_{kinetic}(\sigma) $$

*   **Logic**: We introduced a **Curriculum Parameter** $\alpha \in [0, 1]$.
    *   $\sigma(\alpha)$: The width of the reward Gaussians.
    *   **Start ($\alpha=0$)**: Wide Basin ($\sigma \approx 5.0$). Agent gets paid for being "roughly up".
    *   **End ($\alpha=1$)**: Narrow Basin ($\sigma \approx 1.0$). Agent must be precise.
*   **Result**: **Robust Stabilization**.
    *   The agent learns the general policy early and refines it as the tolerance tightens.

## Iteration 7: Derivative-Based Reward Shaping (Failed)
**Concept**: Use Potential-Based Reward Shaping ($F = \gamma \Phi(s') - \Phi(s)$) to provide a gradient towards the goal.
**Outcome**: **Failed (Orbiting)**. The agent learned to "propeller" the pendulum. It maximized the shaping reward by constantly visiting the high-potential state but never stopping. The velocity penalty was insufficient to arrest the motion.

## Iteration 8: Gaussian Energy Targeting (Current)
**Concept**: Instead of penalizing velocity (which hurts swing-up), we reward matching the **Target Total Energy** of the upright state.
**Formula**:
$$ R = \exp\left( - \frac{(E_{total} - E_{target})^2}{\sigma_E^2} \right) \cdot \exp\left( - \frac{x^2}{\sigma_x^2} \right) $$
**Logic**:
*   **At Bottom**: To match $E_{target}$, the agent *must* have high Kinetic Energy. **Encourages Swing-Up.**
*   **At Top**: To match $E_{target}$, the agent *must* have zero Kinetic Energy. **Encourages Stopping.**
**Curriculum**: $\sigma_E$ anneals from 10.0 (Wide Basin) to 0.5 (Tight Basin).

## Iteration 9: Survival with Penalty (Current)
**Goal**: Learn fine motor skills for stabilization (Reverse Curriculum).
**Strategy**: Start the agent in the **Up-Up** state (Reverse Curriculum) and punish it for falling.
**Formula**:
$$ R = 20.0 - (1.0 \cdot \theta_{err}^2 + 0.1 \cdot \dot{\theta}^2 + 0.1 \cdot x^2 + 0.001 \cdot u^2) $$
**Logic**:
# Reward Function Evolution

This document chronicles the iterative development of the reward function used to solve the Double Pendulum Stabilization task. Each iteration addressed specific failure modes observed during training.

## Iteration 1: The Naive Geometric Reward
**Goal**: Tell the agent "Up is Good".

$$ R = -(\theta_1 - \pi)^2 - (\theta_2 - \pi)^2 - 0.1 x^2 $$

*   **Logic**: Simple squared error penalty on angles and cart position.
*   **Result**: **Failure**.
    *   The agent learned to keep the cart centered but failed to swing up.
    *   The "gradient" of this reward is too shallow when the poles are hanging down. The agent didn't know *how* to get up.

## Iteration 2: The Energy Shaping Reward
**Goal**: Teach the agent to "Pump Energy".

$$ R_{energy} = \exp(-|E_{current} - E_{target}|) $$

*   **Logic**: The upright equilibrium has a specific total energy $E_{target}$. We reward the agent for having this energy, regardless of its position.
*   **Result**: **Partial Success**.
    *   The agent learned to swing up vigorously!
    *   **Problem**: The agent would reach the top but fly right past it. It learned to "orbit" the equilibrium, maintaining constant energy but never stopping.

## Iteration 3: Kinetic Damping
**Goal**: Teach the agent to "Stop at the Top".

$$ R_{kinetic} = \exp(-T) $$

*   **Logic**: Penalize Kinetic Energy ($T$) to encourage stillness.
*   **Result**: **Failure**.
    *   The agent learned to simply stop the pendulum at the bottom (Down-Down).
    *   Minimizing $T$ everywhere discourages the high-velocity swing-up needed to reach the top.

## Iteration 4: Gated Kinetic Damping
**Goal**: "Swing fast when down, stop when up."

$$ R_{kinetic} = \exp(-T) \cdot \underbrace{\exp(-(\theta - \pi)^2)}_{\text{Gate}} $$

*   **Logic**: Multiply the Kinetic Reward by a "Spatial Gate". The agent only gets paid for being still *if* it is also near the top.
*   **Result**: **Success (Unstable)**.
    *   The agent could stabilize, but often fell into "limit cycles" where it flailed around the top.
    *   The transition from "Swing" to "Stabilize" was too harsh.

## Iteration 5: Hybrid Reward with Curriculum (Current)
**Goal**: Smooth the learning curve.

$$ R_{total} = w_s R_{spatial}(\sigma) + w_e R_{energy}(\sigma) + w_k R_{kinetic}(\sigma) $$

*   **Logic**: We introduced a **Curriculum Parameter** $\alpha \in [0, 1]$.
    *   $\sigma(\alpha)$: The width of the reward Gaussians.
    *   **Start ($\alpha=0$)**: Wide Basin ($\sigma \approx 5.0$). Agent gets paid for being "roughly up".
    *   **End ($\alpha=1$)**: Narrow Basin ($\sigma \approx 1.0$). Agent must be precise.
*   **Result**: **Robust Stabilization**.
    *   The agent learns the general policy early and refines it as the tolerance tightens.

## Iteration 7: Derivative-Based Reward Shaping (Failed)
**Concept**: Use Potential-Based Reward Shaping ($F = \gamma \Phi(s') - \Phi(s)$) to provide a gradient towards the goal.
**Outcome**: **Failed (Orbiting)**. The agent learned to "propeller" the pendulum. It maximized the shaping reward by constantly visiting the high-potential state but never stopping. The velocity penalty was insufficient to arrest the motion.

## Iteration 8: Gaussian Energy Targeting (Current)
**Concept**: Instead of penalizing velocity (which hurts swing-up), we reward matching the **Target Total Energy** of the upright state.
**Formula**:
$$ R = \exp\left( - \frac{(E_{total} - E_{target})^2}{\sigma_E^2} \right) \cdot \exp\left( - \frac{x^2}{\sigma_x^2} \right) $$
**Logic**:
*   **At Bottom**: To match $E_{target}$, the agent *must* have high Kinetic Energy. **Encourages Swing-Up.**
*   **At Top**: To match $E_{target}$, the agent *must* have zero Kinetic Energy. **Encourages Stopping.**
**Curriculum**: $\sigma_E$ anneals from 10.0 (Wide Basin) to 0.5 (Tight Basin).

## Iteration 9: Survival with Penalty (Current)
**Goal**: Learn fine motor skills for stabilization (Reverse Curriculum).
**Strategy**: Start the agent in the **Up-Up** state (Reverse Curriculum) and punish it for falling.
**Formula**:
$$ R = 20.0 - (1.0 \cdot \theta_{err}^2 + 0.1 \cdot \dot{\theta}^2 + 0.1 \cdot x^2 + 0.001 \cdot u^2) $$
**Logic**:
*   **Alive Bonus (+20.0)**: Strong incentive to keep the episode running.
*   **Angle Penalty (1.0)**: Moderate penalty for deviating from Upright.
*   **Velocity Penalty (0.1)**: Moderate penalty for moving fast.
*   **Effort Penalty**: Slight penalty for using too much force.
**Outcome**: This is a classic Control Theory cost function (LQR-style) inverted into a reward. By starting "Up", the agent learns stabilization immediately without needing to solve the Swing-Up problem first.

### Iteration 9.1: Sharpened Survival (Current)
**Goal**: Force precise balancing by making "barely surviving" painful.
**Changes**:
*   **Angle Penalty**: Increased from 1.0 to **30.0**.
*   **Velocity Penalty**: Increased from 0.1 to **1.0**.
*   **Physics**: `dt` reduced to 0.005s (200Hz). Termination at 45 degrees.
**Formula**:
$$ R = 20.0 - (30.0 \cdot \theta_{err}^2 + 1.0 \cdot \dot{\theta}^2 + 0.1 \cdot x^2 + 0.001 \cdot u^2) $$
**Logic**: At 45 degrees (0.78 rad), the penalty is $30 \cdot 0.78^2 \approx 18$, canceling the Alive Bonus. The agent *must* be upright to profit.

## Iteration 10: Goal-Conditioned (Phase 6)(Phase 5)
**Goal**: Control multiple equilibria.
$$ R_{dynamic} = f(State, Goal_{target}) $$
*   **Logic**: The formula remains the same as Iteration 9, but the **Target Reference** ($E_{target}, \theta_{target}$) changes dynamically based on the user's command (e.g., "Go to Down-Up").
