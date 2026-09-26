# Project Development Plan
This document outlines the planned phases and tasks for developing Double Pendulum Stabilization.

**Status (2026-09-26):** the double pendulum is swung up from hanging and
balanced at full physics by a designed controller (Phase R: trajectory
optimisation + TVLQR + LQR; 100/100 episodes at $\delta = 1$). No RL policy
balances yet. The chronological log is `docs/EXPERIMENTS.md`; the resume
point is `docs/NEXT_STEPS.md`.

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
    - [completed] From-scratch PyTorch implementation (`src/agent/ppo.py`); the original `stable-baselines3` plan was dropped.
5.  [completed] Reward Function Engineering.
    - [completed] Output: `src/utils/visualizer.py`.

## Phase 4: Robustness & Perturbations
6.  [completed] Perturbation Mechanism.
    - [completed] Allow user to apply impulsive forces.
    - [completed] Simulate continuous wind.
7.  [in-progress] Stress Testing.
    - [partial] LQR basin probe (2026-07-12): ~79 % capture from simultaneous ±14° pole errors, ±0.5 rad/s, ±1 m.
    - [partial] Designed swing-up: 30/30 at 5× wind ($\sigma_w = 5$ N); `tools/eval_swingup.py --wind`.
    - [pending] Systematic sweep of maximum recoverable angle/velocity/impulse for the Phase R controller.
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
5.  [blocked] Train to completion (Difficulty 1.0). PPO reaches $\delta = 1$ but never balances; SAC stalls at $\delta \approx 0.34$. Root cause (energy-level swing-up cannot reach the capture set on a chaotic level set) in `docs/NEXT_STEPS.md`. Superseded for the headline task by Phase R; RL continues under Phase S.
6.  [completed] Verify robustness on full physics — for the Phase R controller (`docs/reports/swingup_d1.md`). Pending for any RL policy.

## Phase 6: Multi-Equilibrium Switching
1.  [partial] Create `DoublePendulumGoalEnv` (Goal-Conditioned) — scaffold in `src/env/double_pendulum_goal.py`, untrained.
2.  [pending] Implement Goal-Conditioned Reward.
3.  [pending] Train agent to switch between Down-Down, Up-Up, Down-Up, Up-Down. Model-based alternative now cheap: `optimize_swingup(start=..., goal=...)` between any two equilibria + LQR at each (see Phase T).
4.  [pending] Interactive Control Demo.

## Phase 7: Velocity Control
1.  [completed] Modify Env to use Velocity Control.
2.  [abandoned] Retrain with Velocity Control (High Gain). With $K_p v_{\max} = 10^5$ N per unit action, balance-scale corrections are ~$10^{-4}$ in action space, far below exploration noise. Use `ForceControl` for balance work.

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

## Phase L: Algorithmic Ceiling Break
**Goal**: Push past the ~6.5 % strict-success ceiling that holds across PPO Phases C-K.
1.  [completed] SAC rewrite (`src/agent/sac.py`, `src/train_sac.py`) with state-dependent $\log\sigma$, twin-Q targets, replay buffer, auto-entropy — Phase N in `docs/EXPERIMENTS.md`. Reached 26.21 % whole-episode strict at $\delta = 0.295$; later shown (Phase O) to be transit-and-crash, not balance.
2.  [completed] LQR bootstrap — done for SAC's replay buffer (Phase Q): 0.2 % terminal-1 s strict. Negative result.
3.  [completed] Algorithm-evolve HP campaign (Phase M): null result.

## Phases O–Q: Honest metrics, energy-reward SAC, LQR-bootstrapped SAC (2026-05)
1.  [completed] Phase O: honest balance metrics (survival, terminal-1 s strict, max sustained strict); whole-episode strict retired as a headline.
2.  [completed] Phase P: SAC + `EnergyShapingReward` — survives with poles tumbling near horizontal (reward's true optimum).
3.  [completed] Phase Q: SAC with LQR-filled replay buffer — same tumbling optimum.
4.  [pending] Commit the Phase O–Q code and reports (`src/control/hybrid_controller.py`, `src/control/equilibria.py`, `src/agent/lqr_bootstrap.py`, `tools/eval_lqr.py`, `tools/eval_hybrid.py`, `tools/rollout_trace.py`, `docs/reports/v2_*`) — they exist only in a local working tree; `docs/EXPERIMENTS.md` has no O–Q entries.

## Phase R: Designed Swing-Up (2026-09-24) — headline task solved
**Goal**: Hanging → upright → balance at $\delta = 1$ without RL (`docs/NEXT_STEPS.md` Option A).
1.  [completed] 2026-07-12 diagnostic: plant and LQR verified; energy swing-up never enters the capture band (0/15).
2.  [completed] Direct multiple-shooting swing-up on the simulator's RK4 (`src/control/swingup.py`, `src/control/data/swingup_d1.npz`).
3.  [completed] Discrete TVLQR tracking + LQR balance + settle phase (`SwingUpController`).
4.  [completed] Evaluation harness + report (`tools/eval_swingup.py`, `docs/reports/swingup_d1.md`): 100/100 episodes, terminal-1 s strict 100 %, steady-state 0.038° / 0.016°, peak force 27 N.
5.  [completed] Fix `src/run_lqr.py` (crashed on `env.force_mag`; drove the env through `VelocityControl`) — now the interactive swing-up demo.
6.  [completed] Tests (`tests/test_swingup.py`).

## Phase S: RL That Balances (next)
**Goal**: First learned policy that genuinely balances (`docs/NEXT_STEPS.md` Option B).
1.  [pending] `--init_mode basin --init_radius r` reset option; SAC with `ForceControl`.
2.  [pending] Milestone: ≥ 95 % terminal-1 s strict at $r = 0.25$ rad, $\delta = 1$.
3.  [pending] Reverse curriculum toward hanging, seeding resets from states along the Phase R trajectory (latest first).
4.  [pending] Optional capture-dwell reward rider (Option C).

## Phase T: Model-Based Equilibrium Switching
1.  [pending] Trajectories between all pairs of the four equilibria via `optimize_swingup(start, goal)`.
2.  [pending] LQR at each unstable equilibrium (down-up, up-down, up-up).
3.  [pending] Interactive switching demo (extends `src/run_lqr.py`).

## Housekeeping
1.  [pending] `tests/test_pipeline_equivalence.py` SHA-256 trajectory baselines fail on this machine's numpy/LAPACK build (3 tests, pre-existing, env code untouched). Make them platform-tolerant or document the pinned build.
2.  [pending] Phase 8.5 single-pendulum smoke run.
