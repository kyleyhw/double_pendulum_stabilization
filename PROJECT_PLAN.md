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
    - [completed] Algorithm: Proximal Policy Optimization (PPO).
    - [completed] Library: `stable-baselines3`.
5.  [completed] Reward Function Engineering.
    - [completed] Output: `src/utils/visualizer.py`.

## Phase 4: Robustness & Perturbations
6.  [completed] Perturbation Mechanism.
    - [completed] Allow user to apply impulsive forces.
    - [completed] Simulate continuous wind.
7.  [pending] Stress Testing.
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

### Tasks
1.  [completed] Implement `DoublePendulumCartEnv` with variable physics ($g$, friction).
2.  [completed] Implement `set_curriculum(difficulty)` method.
3.  [completed] Implement **Exponential Continuity Reward**.
4.  [completed] Implement **Ratchet Curriculum** in `train.py`.
5.  [in-progress] Train to completion (Difficulty 1.0).
6.  [pending] Verify robustness on full physics.

## Phase 6: Multi-Equilibrium Switching
1.  [pending] Create `DoublePendulumGoalEnv` (Goal-Conditioned).
2.  [pending] Implement Goal-Conditioned Reward.
3.  [pending] Train agent to switch between Down-Down, Up-Up, Down-Up, Up-Down.
4.  [pending] Interactive Control Demo.

## Phase 7: Velocity Control
1.  [completed] Modify Env to use Velocity Control.
2.  [paused] Retrain with Velocity Control (High Gain).

## Phase 8: Modularization & Single Pendulum
1.  [completed] Create `src/strategies/controls.py` & `rewards.py`.
2.  [completed] Refactor `DoublePendulumEnv` to use strategies (subclass of `CartPendulumBase`).
3.  [completed] Implement `SinglePendulumEnv` using strategies.
4.  [completed] Update `train.py` with `--env`, `--control`, `--reward` args.
5.  [pending] Verify Single Pendulum Training (smoke run on the optimised pipeline).

## Phase K: Pipeline Runtime Optimisation
**Goal**: Reduce wall-time of the training+test pipeline so further algorithmic experiments (SAC, LQR-bootstrap) become tractable. Hard constraint: physics bit-identical to master baseline.
1.  [completed] Build agent-evolve infrastructure (`tools/evolve_eval.py`, `tests/test_pipeline_equivalence.py`, `agent-evolve.yaml`).
2.  [completed] Run baseline (master): 6431 ms / PPO update at `--n_envs 4 --rollout_steps 256`.
3.  [completed] R1 dispatch 3 explore candidates in parallel git worktrees (env-only, trainer-only, full-stack).
4.  [completed] Score + review candidates; full-stack wins with all 20 tests passing.
5.  [completed] Open and merge PR #1 (5.77x speedup, bit-equivalent).
6.  [completed] Re-train Phase I best on optimised pipeline; confirm policy quality matches (4.3 % strict at $\delta = 0.445$, vs 4.7 % parent — within noise).
7.  [completed] R2 dispatch 3 mutate candidates (Cramer's rule on the 3x3 solve, numba @njit on `_dynamics`, batched dynamics across N envs).
8.  [completed] R2 score + review: c6 (batched dynamics) wins at 1.43x over c3, 21/21 tests still pass per-row bit-identical.
9.  [completed] Open and merge PR #2 (additional 1.43x → ~8.25x cumulative vs original 6431 ms baseline).
10. [completed] Re-train Phase K best on c6 pipeline; confirm policy quality unchanged (4.4 % strict at $\delta = 0.465$, vs Phase K parent 5.3 % — within 30-ep stochastic noise).

## Phase L: Algorithmic Ceiling Break (next)
**Goal**: Push past the ~6.5 % strict-success ceiling that holds across PPO Phases C-K. See `docs/NEXT_STEPS.md` for ranked options.
1.  [pending] Option A: SAC rewrite (`src/agent/sac.py`) with state-dependent $\log\sigma$, twin-Q targets, replay buffer, auto-entropy.
2.  [pending] Option B (alternative): LQR behavioural-cloning bootstrap of the existing PPO actor.
3.  [pending] Validate on `--env single` smoke run, then `--env double` from scratch.
