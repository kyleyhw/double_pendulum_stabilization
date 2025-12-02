# Phase 5: Multi-Equilibrium Control (Goal-Conditioned RL)

## 1. The Objective
The Double Pendulum has four primary equilibria:
1.  **Down-Down** (Stable): Both poles hanging down.
2.  **Up-Up** (Unstable): Both poles balanced up.
3.  **Down-Up** (Unstable): Pole 1 down, Pole 2 balanced up.
4.  **Up-Down** (Unstable): Pole 1 balanced up, Pole 2 hanging down.

Our goal is to train a **single agent** that can transition between any of these states on command.

## 2. The Strategy: Goal-Conditioned RL
Instead of training 4 separate agents, we train one agent with an augmented observation space.

### 2.1 Augmented State
The input to the Neural Network is expanded:
$$ State_{augmented} = [State_{physics} (6), Goal_{onehot} (4)] $$
*   **Input Dimension**: $6 + 4 = 10$.
*   **Goal Vector**: A one-hot encoding of the target mode (e.g., `[0, 1, 0, 0]` for Up-Up).

### 2.2 Dynamic Reward Function
The reward function structure remains the same (Hybrid Physical Reward), but the **Target Reference** changes dynamically based on the Goal Vector.

**Algorithm:**
1.  **Read Goal**: Determine target angles ($\theta_{t1}, \theta_{t2}$) and target energy ($E_{target}$) for the current mode.
2.  **Compute Error**: Calculate spatial and energy error relative to *that specific target*.
3.  **Compute Reward**: $R = \text{HybridReward}(Error_{dynamic})$.

This allows the agent to learn a generalized control policy: "Minimize error relative to Input Goal".

## 3. Implementation Details

### 3.1 Environment (`DoublePendulumGoalEnv`)
*   Inherits from `DoublePendulumCartEnv`.
*   Adds `self.target_mode` state.
*   **Reset**: Randomly selects a target mode to ensure the agent practices all transitions.

### 3.2 Training
*   **Curriculum**: We reuse the Reward Annealing ($\alpha$) from Phase 4 to help the agent learn the unstable equilibria.
*   **Randomization**: By constantly switching targets during training, the agent learns to be robust to state transitions.

### 3.3 Interactive Simulation
*   **Keys**: `1`, `2`, `3`, `4` map to the four equilibria.
*   **Real-Time**: The user can change the goal mid-simulation, forcing the agent to swing the pendulum from one configuration to another dynamically.
