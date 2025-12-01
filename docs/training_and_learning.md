# Training and Learning Documentation

## 1. Reinforcement Learning Framework

We model the double pendulum stabilization as a **Markov Decision Process (MDP)** defined by the tuple $(S, A, P, R, \gamma)$:

*   **State Space ($S$)**: Continuous, 8-dimensional.

    $$
    s = [x, \sin\theta_1, \cos\theta_1, \sin\theta_2, \cos\theta_2, \dot{x}, \dot{\theta}_1, \dot{\theta}_2]
    $$

    We use trigonometric encoding for angles to avoid discontinuities at $\pm \pi$.
*   **Action Space ($A$)**: Continuous, 1-dimensional.

    $$
    a \in [-1, 1]
    $$

    Represents the normalized force applied to the cart. The actual force is scaled by $F_{max}$.
*   **Dynamics ($P$)**: Deterministic physics governed by the Lagrangian equations of motion (see `physics_derivation.md`).
*   **Reward Function ($R$)**: Dense, shaped reward to guide the agent towards the unstable equilibrium.
*   **Discount Factor ($\gamma$)**: $0.99$, encouraging long-term stability.
*   **Simulation Parameters**:
    *   Timestep ($dt$): $0.02$ s
    *   Max Force ($F_{max}$): $20.0$ N
    *   Max Episode Length: $2000$ steps ($40$ s)

## 2. Proximal Policy Optimization (PPO)

We use **PPO-Clip**, a policy gradient algorithm that optimizes a surrogate objective function to ensure stable policy updates.

### 2.1 Objective Function
The PPO objective is designed to maximize the expected return while limiting the change in the policy distribution between updates.


$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]
$$


Where:
*   $\pi_\theta(a_t|s_t)$ is the current policy.
*   $\pi_{\theta_{old}}(a_t|s_t)$ is the policy before the update.
*   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.
*   $\hat{A}_t$ is the **Generalized Advantage Estimation (GAE)**.
*   $\epsilon$ is the clipping parameter (typically $0.2$), defining the "trust region".

### 2.2 Total Loss
The final loss function minimized by the optimizer includes the value function error and an entropy bonus for exploration:


$$
L_{total} = -L^{CLIP}(\theta) + c_1 L^{VF}(\theta) - c_2 S[\pi_\theta](s_t)
$$


*   $L^{VF}(\theta) = (V_\theta(s_t) - V_t^{target})^2$: Mean Squared Error of the value predictor.
*   $S[\pi_\theta]$: Entropy of the policy distribution. Maximizing entropy prevents premature convergence to suboptimal deterministic policies.
*   $c_1, c_2$: Coefficients weighting the value loss ($0.5$) and entropy bonus ($0.01$).

## 3. Reward Function Design

The reward function is the critical signal that tells the agent what "success" looks like. We use a **Positive Gaussian Reward** formulation.
> [!IMPORTANT]
> **Why Positive?**
> A previous iteration used a negative quadratic cost ($-x^2$). This led to the **"Suicide Bug"**, where the agent learned to crash the cart immediately to end the episode and avoid accumulating further negative rewards.
> By using a positive reward $R \in (0, 1]$, staying alive is always better than crashing (which yields 0 future reward).

$$
r_t = \exp \left( - \left( w_{\theta_1} \tilde{\theta}_1^2 + w_{\theta_2} \tilde{\theta}_2^2 + w_x x^2 + w_{\dot{q}} \|\dot{q}\|^2 \right) \right)
$$


### Terms:
1.  **Angle Errors ($\tilde{\theta}_i$)**:

    $$
    \tilde{\theta}_i = \text{normalize}(\theta_i - \pi)
    $$

    Measures the deviation from the upright position ($\pi$). We want this to be 0.
2.  **Cart Position ($x$)**:
    Penalizes the cart for moving away from the center of the track ($x=0$). This prevents the agent from "cheating" by running off to infinity to balance the pendulum.
3.  **Angular Velocities ($\dot{q}$)**:
    Penalizes high velocities. Stability implies $\dot{\theta} \approx 0$.
4.  **Implicit Survival Incentive**:
    Since $r_t > 0$ always, the agent maximizes return by keeping the episode going as long as possible.

### Weights:
*   $w_{\theta_1} = 1.0$: High priority on the first link.
*   $w_{\theta_2} = 1.0$: High priority on the second link.
*   $w_x = 0.5$: **Increased Priority** on cart position to encourage re-centering and prevent wall crashes.
*   $w_{\dot{q}} = 0.01$: Small penalty to dampen oscillations.

## 4. Neural Network Architecture

We use a shared backbone with separate heads for the Actor and Critic (or separate networks depending on implementation specifics).

### Actor (Policy Network)
*   **Input**: State vector (size 8).
*   **Hidden Layers**: 2 layers of 64 units with `Tanh` activation.
*   **Output**:
    *   Mean ($\mu$): `Tanh` activation (mapped to $[-1, 1]$).
    *   Log Standard Deviation ($\log \sigma$): Learnable parameter (state-independent) or output of a layer.
*   **Action Selection**: $a_t \sim \mathcal{N}(\mu(s_t), \sigma)$.

### Critic (Value Network)
*   **Input**: State vector (size 8).
*   **Hidden Layers**: 2 layers of 64 units with `Tanh` activation.
*   **Output**: Scalar value $V(s_t)$ (linear activation). Estimate of discounted future return.

## 5. Exploration Strategy

### 5.1 Zero-Mean Initialization
We initialize the final layer of the Actor network with very small weights and zero bias.
*   **Effect**: The initial policy $\pi(a|s)$ is a Gaussian with mean $\approx 0$.
*   **Why**: Random initialization often leads to a non-zero mean (e.g., always pushing right). This biases exploration and can cause the agent to get stuck in a local optimum of "crashing right". Zero-mean ensures unbiased "wiggling" (Left/Right) at the start.

### 5.2 Perturbative Exploration
To ensure the agent learns to recover from disturbances, we employ:
*   **Wind**: Continuous Gaussian noise added to the force action during training.
*   **Impulses**: Instantaneous forces applied to the cart.
This forces the agent to actively correct deviations rather than memorizing a single stable trajectory.

## 6. Training Loop

1.  **Collection Phase**: Run the agent in the environment for $N$ steps (e.g., 2048), collecting trajectories $(s_t, a_t, r_t, s_{t+1})$.
2.  **Advantage Calculation**: Compute GAE using the collected rewards and value estimates.
3.  **Update Phase**:
    *   Shuffle the collected data.
    *   Iterate through mini-batches.
    *   Perform gradient descent on $L_{total}$ for $K$ epochs (e.g., 10).
4.  **Repeat**: Continue until max episodes or convergence.
