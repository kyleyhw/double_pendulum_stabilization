# Double Pendulum Stabilization with Reinforcement Learning

## 1. Project Objective
The goal of this project is to stabilize a **double pendulum on a cart** in the unstable upright equilibrium position ($\theta_1 = \pi, \theta_2 = \pi$) using Deep Reinforcement Learning (PPO). This system is a classic benchmark in control theory due to its high nonlinearity and chaotic dynamics.

![Visualizer Screenshot](docs/images/visualizer_screenshot.png)

## 2. Training Methodology
We use **Proximal Policy Optimization (PPO)**, a state-of-the-art Deep Reinforcement Learning algorithm.

![Learning Curve](docs/images/learning_curve.png)

*   **Architecture**: The agent uses a **Neural Network** (Multi-Layer Perceptron) with two heads:
    *   **Actor (Policy)**: Outputs the mean and standard deviation of a Gaussian distribution for the action (force). This allows the agent to "explore" the state space stochastically.
    *   **Critic (Value)**: Estimates the expected future reward from the current state, used to compute the "advantage" of an action.
*   **Exploration**:
    *   **Stochastic Policy**: The agent samples actions from a Gaussian distribution.
    *   **Zero-Mean Initialization**: The policy is initialized to be unbiased (zero mean), ensuring the agent explores both directions (Left/Right) equally at the start.
    *   **Perturbations**: We can introduce **Wind** (continuous noise) and **Impulses** (sudden pushes) to force the agent to learn robust stabilization strategies.

## 3. Mathematical Formulation

### 3.1 System Dynamics
The system consists of a cart of mass $M$ moving on a 1D track, with two links of mass $m_1, m_2$ and length $l_1, l_2$.

**Generalized Coordinates**:

$$
q = [x, \theta_1, \theta_2]^T
$$

where $x$ is the cart position, and $\theta_i$ are the angles from the vertical down position.

**Lagrangian Mechanics**:
The Equations of Motion (EOM) are derived from the Euler-Lagrange equation:

$$
\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{q}} \right) - \frac{\partial \mathcal{L}}{\partial q} = \tau
$$

where $\mathcal{L} = T - V$.

This yields the standard robotic manipulator form:

$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = B u
$$


*   **Inertia Matrix** $M(q)$: Symmetric, positive-definite matrix encoding the mass distribution and coupling between links.
*   **Coriolis & Centrifugal Matrix** $C(q, \dot{q})$: Contains terms like $\dot{\theta}_i^2$ and $\dot{\theta}_1 \dot{\theta}_2$.
*   **Gravity Vector** $G(q)$: Derived from potential energy $V(\theta)$.
*   **Control Input** $u$: The force $F$ applied to the cart.

For the full derivation, see [docs/physics_derivation.md](docs/physics_derivation.md).

### 3.2 Controllability Analysis
Before training, we mathematically verified that the unstable equilibria (Up-Up, Down-Up, Up-Down) are **controllable**.
Using the Kalman Rank Condition on the linearized dynamics, we proved that the system is fully controllable ($\text{rank}(\mathcal{C}) = 6$) despite being underactuated.

See the full proof and analysis in [docs/controllability_analysis.md](docs/controllability_analysis.md).

### 3.3 Stabilization Strategy (Phase 4)
To achieve robust stabilization, we used **Energy Shaping** and **Curriculum Learning**.
*   **Energy Shaping**: Rewards the agent for having the correct total energy, creating a global gradient.
*   **Curriculum**: Slowly tightens the reward tolerance ($\sigma$) from "Wide" to "Narrow" during training.
*   **Details**: [docs/stabilization_strategy.md](docs/stabilization_strategy.md)

### 3.4 Multi-Equilibrium Strategy (Phase 5)
To control all 4 equilibria with one agent, we use **Goal-Conditioned RL**.
*   **Augmented State**: The agent receives the target equilibrium ID as an input.
*   **Dynamic Reward**: The reward target shifts based on the requested goal.
*   **Details**: [docs/multi_equilibrium_strategy.md](docs/multi_equilibrium_strategy.md)

### 3.5 Reward Evolution
For a detailed history of how we derived the reward function (from Naive Geometric to Hybrid Curriculum), see:
*   [docs/reward_history.md](docs/reward_history.md)

### 3.6 State Space
The RL agent observes the full state vector:

$$
\mathbf{s} = [x, \sin\theta_1, \cos\theta_1, \sin\theta_2, \cos\theta_2, \dot{x}, \dot{\theta}_1, \dot{\theta}_2]
$$

*Note: We use $\sin/\cos$ of angles to avoid discontinuity at $\pm \pi$.*

### 3.4 Reward Function (Hybrid Physical Reward)
The reward function is a weighted sum of three physical components:

$$
R_{total} = w_{spatial} R_{spatial} + w_{energy} R_{energy} + w_{kinetic} R_{kinetic}
$$

1.  **Spatial Reward ($R_{spatial}$)**: Gaussian penalty on position and angle errors. Encourages the agent to be in the correct configuration.
2.  **Energy Reward ($R_{energy}$)**: Penalizes deviation from the target total energy ($E_{target}$). Encourages the agent to pump/dissipate energy to reach the correct manifold.
3.  **Kinetic Damping ($R_{kinetic}$)**: Penalizes kinetic energy ($T$) when near the target. Forces the agent to stabilize (stop moving) once it reaches the upright position.

This "Physical Reward" structure guides the agent through the energy landscape rather than just minimizing geometric error.


## 4. Project Structure & Separation
The project is strictly separated into **Simulation (Physics)** and **Agent (Brain)**:

```
double_pendulum_stabilization/
├── src/
│   ├── env/                # SIMULATION (The World)
│   │   └── double_pendulum.py  # Physics engine, EOMs, RK4 integrator.
│   ├── agent/              # AGENT (The Brain)
│   │   └── ppo.py              # Neural Network architecture & PPO algorithm.
│   ├── train.py            # TRAINING (The School)
│   │   └── ...                 # Connects Env and Agent for learning.
│   └── simulate.py         # INFERENCE (The Test)
│       └── ...                 # Runs the Agent in the Env without learning.
├── docs/                   # Documentation
│   ├── physics_derivation.md
│   └── visualization.md
└── ...
```

## 5. Usage

### Stabilization Strategy: "The Ratchet"
We use a **Curriculum Learning** approach inspired by Underactuated Robotics [[3]](#references).

1.  **Variable Physics**: The agent starts in a "Toy Universe" (Low Gravity, High Friction) and graduates to the "Real World" (Standard Gravity, Zero Friction).
2.  **The Ratchet**: Difficulty increases by **1%** only when the agent beats its **all-time best** average reward.
3.  **Exponential Continuity Reward**: We reward the agent exponentially for the duration it keeps the poles upright ($R \propto e^t$).
    *   Unlike a simple "step reward" (which encourages surviving *just* long enough), this creates a massive incentive for **infinite** stability.
    *   $R_t = (\exp(t_{upright}) - 1) \times \text{penalty}$
4.  **Stagnation Jiggle (Dynamic Decorrelation)**:
    *   If the agent gets stuck at a specific difficulty level (reward plateau), we linearly increase the exploration noise (`min_std`).
    *   This forces the agent to break out of local optima and find robust solutions that work across the curriculum.
    *   The noise resets immediately upon leveling up.

See [docs/stabilization_strategy.md](docs/stabilization_strategy.md) for full details.

## Installation
```bash
git clone https://github.com/kyleyhw/double_pendulum_stabilization.git
cd double_pendulum_stabilization
python -m venv venv
# Activate venv (Windows: venv\Scripts\activate, Unix: source venv/bin/activate)
pip install -r requirements.txt
```

### Running
*   **Train**: `python src/train.py`
*   **Generate Report**: `python src/generate_report.py`
    *   Generates `overlay_montage.mp4`, `final_run.mp4`, `comparison.mp4`, and `learning_curve.png`.
    *   Automatically handles random seeds and timestamping.
*   **Visualize Single Run**: `python src/simulate.py`

**Progress Montage**:
To see the agent's improvement over time (requires multiple checkpoints in `logs/`):
```bash
# Generate full report (Overlay + Final Run + Comparison)
python src/generate_report.py
```

### Robustness Testing
You can test the agent's stability against external forces (wind, pushes).
See [docs/robustness.md](docs/robustness.md) for details.

```bash
# Run with wind
python src/simulate.py --wind 2.0

# Use Left/Right Arrow keys to push the cart during simulation.
```

## References
1.  [OpenAI Gym Documentation](https://www.gymlibrary.dev/)
2.  [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
3.  [Underactuated Robotics (Tedrake)](http://underactuated.mit.edu/)
4.  [Double Pendulum Chaos (YouTube)](https://www.youtube.com/watch?v=pWekXMZJ2zM)
5.  [Russ Tedrake: Underactuated Robotics (YouTube)](https://www.youtube.com/watch?v=9gQQAO4I1Ck&t=675s)
