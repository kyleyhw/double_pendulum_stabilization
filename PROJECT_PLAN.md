# Project Development Plan
This document outlines the planned phases and tasks for developing Double Pendulum Stabilization.

## Phase 1: Mathematical Foundation & Physics Engine
1.  [completed] Derivation of Equations of Motion (EOM).
    - [completed] Use Lagrangian Mechanics ($L = T - V$).
    - [completed] Define generalized coordinates: $q = [x, \theta_1, \theta_2]$.
    - [completed] Derive the coupled system of differential equations.
    - [completed] Output: `docs/physics_derivation.md` (with LaTeX).
2.  [completed] Environment Implementation.
    - [completed] Create `DoublePendulumCartEnv` inheriting from `gymnasium.Env`.
    - [completed] Implement `step()` using Runge-Kutta (RK4) integration for precision.
    - [completed] Output: `src/env/double_pendulum.py`.
3.  [completed] Verification.
    - [completed] Test energy conservation (in frictionless setting).
    - [completed] Verify behavior at limits (single pendulum limits).
    - [completed] Output: `tests/test_physics.py`.

## Phase 2: Reinforcement Learning Implementation
4.  [completed] Agent Setup.
    - [completed] Algorithm: Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC).
    - [completed] Library: `stable-baselines3` (as a baseline) or custom implementation.
5.  [completed] Reward Function Engineering.
    - [completed] Output: `src/utils/visualizer.py`.

## Phase 4: Robustness & Perturbations
9.  [completed] Perturbation Mechanism.
    - [completed] Allow user to apply impulsive forces (mouse click or key press).
    - [completed] Simulate continuous wind (random noise added to state/force).
10. [pending] Stress Testing.
    - [pending] Quantify the maximum recoverable angle/velocity.
    - [pending] Output: `docs/robustness_report.md`.

## Phase 5: Advanced Control - Multi-Equilibrium Cycling
# Project Development Plan
This document outlines the planned phases and tasks for developing Double Pendulum Stabilization.

## Phase 1: Mathematical Foundation & Physics Engine
1.  [completed] Derivation of Equations of Motion (EOM).
    - [completed] Use Lagrangian Mechanics ($L = T - V$).
    - [completed] Define generalized coordinates: $q = [x, \theta_1, \theta_2]$.
    - [completed] Derive the coupled system of differential equations.
    - [completed] Output: `docs/physics_derivation.md` (with LaTeX).
2.  [completed] Environment Implementation.
    - [completed] Create `DoublePendulumCartEnv` inheriting from `gymnasium.Env`.
    - [completed] Implement `step()` using Runge-Kutta (RK4) integration for precision.
    - [completed] Output: `src/env/double_pendulum.py`.
3.  [completed] Verification.
    - [completed] Test energy conservation (in frictionless setting).
    - [completed] Verify behavior at limits (single pendulum limits).
    - [completed] Output: `tests/test_physics.py`.

## Phase 2: Reinforcement Learning Implementation
4.  [completed] Agent Setup.
    - [completed] Algorithm: Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC).
    - [completed] Library: `stable-baselines3` (as a baseline) or custom implementation.
5.  [completed] Reward Function Engineering.
    - [completed] Output: `src/utils/visualizer.py`.

## Phase 4: Robustness & Perturbations
9.  [completed] Perturbation Mechanism.
    - [completed] Allow user to apply impulsive forces (mouse click or key press).
    - [completed] Simulate continuous wind (random noise added to state/force).
10. [pending] Stress Testing.
    - [pending] Quantify the maximum recoverable angle/velocity.
    - [pending] Output: `docs/robustness_report.md`.
## Phase 5: Curriculum Learning & Robust Stabilization
**Goal**: Achieve robust swing-up and stabilization by gradually increasing physics difficulty.

### Strategy: "The Ratchet"
*   **Concept**: Start with a "toy universe" (Low Gravity, High Friction) and ratchet up difficulty only when the agent proves mastery.
*   **Curriculum**:
    *   **Gravity**: $2.0 \to 9.81 m/s^2$.
    *   **Friction**: $0.5 \to 0.0$ (Cart), $0.1 \to 0.0$ (Pole).
    *   **Reward Threshold**: $90^\circ \to 10^\circ$.
*   **Adaptation Logic**:
    *   Increase difficulty by **1%** (0.01) *only* if `avg_reward > best_avg_reward` (All-time High).
    *   This ensures the agent never advances prematurely.
*   **Reward Function**:
    *   **Exponential Continuity**: $R_t = \exp(\text{time\_above\_threshold}) - 1$.
    *   Incentivizes long, unbroken periods of stabilization.

### References
*   **Inspiration**: [Underactuated Robotics (YouTube)](https://www.youtube.com/watch?v=9gQQAO4I1Ck&t=675s) - Concepts on energy shaping and stabilization.

### Tasks
1.  [x] Implement `DoublePendulumCartEnv` with variable physics ($g$, friction).
2.  [x] Implement `set_curriculum(difficulty)` method.
3.  [x] Implement **Exponential Continuity Reward**.
4.  [x] Implement **Ratchet Curriculum** in `train.py`.
5.  [ ] Train to completion (Difficulty 1.0).
6.  [ ] Verify robustness on full physics.

## Phase 6: Multi-Equilibrium Switching
1.  Create `DoublePendulumGoalEnv` (Goal-Conditioned).
2.  Implement Goal-Conditioned Reward.
3.  Train agent to switch between Down-Down, Up-Up, Down-Up, Up-Down.
4.  Interactive Control Demo.
