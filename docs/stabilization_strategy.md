# Stabilization Strategy: The Ratchet Curriculum

## Overview
Stabilizing a double pendulum on a cart from a "Down-Down" position is a challenging control problem due to its chaotic dynamics and underactuated nature. To solve this, we employ a **Curriculum Learning** approach combined with a **Ratchet Mechanism** and **Exponential Rewards**.

## 1. The Physics Curriculum
We define a "Difficulty" parameter $\alpha \in [0, 1]$ that interpolates between a "Toy Universe" and the "Real World".

| Parameter | $\alpha=0$ (Easy) | $\alpha=1$ (Hard) | Rationale |
| :--- | :--- | :--- | :--- |
| **Gravity ($g$)** | $2.0 m/s^2$ | $9.81 m/s^2$ | Lower gravity slows down dynamics, giving the agent more reaction time. |
| **Cart Friction** | $0.5$ | $0.0$ | High friction dampens mistakes and prevents the cart from flying off. |
| **Pole Friction** | $0.1$ | $0.0$ | Joint friction helps stabilize the poles passively. |
| **Reward Threshold** | $90^\circ$ | $10^\circ$ | A wide basin allows the agent to find "roughly up" solutions first. |

## 2. The Ratchet Mechanism
Instead of a fixed schedule, we use an adaptive "Ratchet" to increase difficulty.

*   **Logic**: Difficulty increases by $\Delta \alpha = 0.01$ (1%) *only* if the agent's average reward exceeds its **all-time best** average reward.
*   **Why?**: This ensures the agent has truly mastered the current level before moving on. If performance dips (due to harder physics), the difficulty stays constant until the agent recovers and surpasses its previous peak.

## 3. Exponential Continuity Reward
To encourage robust stabilization (holding the poles up for long periods), we use an exponential reward function based on the duration of continuous stabilization.

$$ R_t = \exp(\text{time\_above\_threshold}) - 1.0 $$

*   **Time Above Threshold**: The continuous time (in seconds) that both poles have been within the `Reward Threshold` of the vertical.
*   **Reset**: If either pole falls out of the threshold, the timer resets to 0.
*   **Effect**: This provides massive incentives for survival. Holding for 5 seconds is exponentially better than holding for 1 second.

## References
*   **Underactuated Robotics**: [YouTube Lecture](https://www.youtube.com/watch?v=9gQQAO4I1Ck&t=675s) - Discusses energy shaping, swing-up control, and stabilization of underactuated systems.
