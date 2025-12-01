# Double Pendulum Stabilization with Reinforcement Learning

## Objective
To simulate a double pendulum on a movable cart and train a reinforcement learning agent to stabilize it in the upright (unstable) equilibrium position. The project emphasizes rigorous physics derivation, "from-scratch" implementation logic, and high-quality visualization of the learning process.

## Mathematical Overview
The system consists of a cart (mass $M$) and two pendulum links (masses $m_1, m_2$, lengths $l_1, l_2$).
The equations of motion are derived using Lagrangian Mechanics:
$$ \mathcal{L} = T - V $$
where $T$ is the kinetic energy and $V$ is the potential energy.
The state space is $\mathbf{s} = [x, \dot{x}, \theta_1, \dot{\theta}_1, \theta_2, \dot{\theta}_2]^T$.
The action space is the force $F$ applied to the cart.

## Project Structure
```
double_pendulum_stabilization/
├── docs/                   # Documentation and Math Derivations
│   ├── physics_derivation.md
│   └── robustness_report.md
├── src/                    # Source Code
│   ├── env/                # Physics Environment
│   │   └── double_pendulum.py
│   ├── agent/              # RL Agent Implementation
│   └── utils/              # Visualization and Logging
│       └── visualizer.py
├── tests/                  # Unit Tests
│   └── test_physics.py
├── PROJECT_PLAN.md         # Development Plan
└── README.md               # This file
```

## Getting Started
(Instructions to be added after implementation)
