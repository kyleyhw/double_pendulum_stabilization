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
11. [pending] Define Equilibria.
    - [pending] **Down-Down** (Stable, $E_{min}$).
    - [pending] **Up-Up** (Unstable, $E_{max}$).
    - [pending] **Down-Up** (Pole 1 Down, Pole 2 Up).
    - [pending] **Up-Down** (Pole 1 Up, Pole 2 Down).
12. [pending] Trajectory Planning & Control.
    - [pending] Implement a goal-conditioned controller (input: Target State).
    - [pending] Demonstrate transitions between ALL pairs (e.g., Down-Up <-> Up-Down).
    - [pending] Mechanism: Dynamic switching of $E_{target}$ and Spatial Target in the reward function.
