# Physics Derivation: Double Pendulum on a Cart

## 1. System Definition
We consider a system consisting of:
*   A cart of mass $M$ moving on a 1D horizontal track (position $x$).
*   A first pendulum link of mass $m_1$ and length $l_1$ attached to the cart.
*   A second pendulum link of mass $m_2$ and length $l_2$ attached to the end of the first link.

**Generalized Coordinates**:
$$ q = \begin{bmatrix} x \\ \theta_1 \\ \theta_2 \end{bmatrix} $$
where:
*   $x$ is the cart position.
*   $\theta_1$ is the angle of the first link with respect to the vertical (0 = down, $\pi$ = up).
*   $\theta_2$ is the angle of the second link with respect to the vertical.

**Note**: For the RL task, we often define 0 as "up" for reward convenience, but for standard physics derivation, we usually define 0 as "down" (stable equilibrium) or "up" (unstable). Let's define $\theta = 0$ as the **vertical upward** position for this derivation to align with the control goal, or stick to standard convention.
*Standard Convention*: $\theta=0$ is down.
*Control Goal*: Stabilize at $\theta_1 = \pi, \theta_2 = \pi$.
Let's stick to $\theta=0$ is **down** (gravity acts in $+y$ direction if $y$ is down, or $-y$ if $y$ is up).
Let's use a standard Cartesian frame: $y$ points **up**. Gravity $g$ acts in $-y$.
$\theta=0$ corresponds to the pendulum hanging **down**.
Positions:
$$
\begin{aligned}
x_c &= x \\
y_c &= 0 \\
x_1 &= x + l_1 \sin \theta_1 \\
y_1 &= -l_1 \cos \theta_1 \\
x_2 &= x + l_1 \sin \theta_1 + l_2 \sin \theta_2 \\
y_2 &= -l_1 \cos \theta_1 - l_2 \cos \theta_2
\end{aligned}
$$

## 2. Kinematics (Velocities)
Differentiating with respect to time:
$$
\begin{aligned}
\dot{x}_c &= \dot{x} \\
\dot{y}_c &= 0 \\
\dot{x}_1 &= \dot{x} + l_1 \dot{\theta}_1 \cos \theta_1 \\
\dot{y}_1 &= l_1 \dot{\theta}_1 \sin \theta_1 \\
\dot{x}_2 &= \dot{x} + l_1 \dot{\theta}_1 \cos \theta_1 + l_2 \dot{\theta}_2 \cos \theta_2 \\
\dot{y}_2 &= l_1 \dot{\theta}_1 \sin \theta_1 + l_2 \dot{\theta}_2 \sin \theta_2
\end{aligned}
$$

Squared velocities ($v^2 = \dot{x}^2 + \dot{y}^2$):
$$
\begin{aligned}
v_c^2 &= \dot{x}^2 \\
v_1^2 &= (\dot{x} + l_1 \dot{\theta}_1 \cos \theta_1)^2 + (l_1 \dot{\theta}_1 \sin \theta_1)^2 \\
      &= \dot{x}^2 + 2\dot{x}l_1\dot{\theta}_1\cos\theta_1 + l_1^2\dot{\theta}_1^2 \\
v_2^2 &= (\dot{x} + l_1 \dot{\theta}_1 \cos \theta_1 + l_2 \dot{\theta}_2 \cos \theta_2)^2 + (l_1 \dot{\theta}_1 \sin \theta_1 + l_2 \dot{\theta}_2 \sin \theta_2)^2 \\
      &= \dot{x}^2 + l_1^2\dot{\theta}_1^2 + l_2^2\dot{\theta}_2^2 + 2\dot{x}l_1\dot{\theta}_1\cos\theta_1 + 2\dot{x}l_2\dot{\theta}_2\cos\theta_2 + 2l_1l_2\dot{\theta}_1\dot{\theta}_2\cos(\theta_1 - \theta_2)
\end{aligned}
$$

## 3. Energy
**Kinetic Energy ($T$)**:
$$ T = \frac{1}{2}M v_c^2 + \frac{1}{2}m_1 v_1^2 + \frac{1}{2}m_2 v_2^2 $$

**Potential Energy ($V$)**:
$$ V = m_1 g y_1 + m_2 g y_2 = -m_1 g l_1 \cos \theta_1 - m_2 g (l_1 \cos \theta_1 + l_2 \cos \theta_2) $$
$$ V = -(m_1 + m_2) g l_1 \cos \theta_1 - m_2 g l_2 \cos \theta_2 $$

**Lagrangian ($\mathcal{L}$)**:
$$ \mathcal{L} = T - V $$

## 4. Equations of Motion
We solve $\frac{d}{dt} \frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i} = F_i$.
Generalized forces: $F_x = F$ (control force), $F_{\theta_1} = 0$, $F_{\theta_2} = 0$.

This results in a system of the form:
$$ M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q) = \tau $$

### Mass Matrix $M(q)$
$$
M(q) = \begin{bmatrix}
M + m_1 + m_2 & (m_1 + m_2)l_1 \cos \theta_1 & m_2 l_2 \cos \theta_2 \\
(m_1 + m_2)l_1 \cos \theta_1 & (m_1 + m_2)l_1^2 & m_2 l_1 l_2 \cos(\theta_1 - \theta_2) \\
m_2 l_2 \cos \theta_2 & m_2 l_1 l_2 \cos(\theta_1 - \theta_2) & m_2 l_2^2
\end{bmatrix}
$$

### Coriolis & Centrifugal Matrix $C(q, \dot{q})$
Terms involving $\dot{\theta}^2$ and $\dot{\theta}_i \dot{\theta}_j$.
$$
C(q, \dot{q})\dot{q} = \begin{bmatrix}
-(m_1 + m_2)l_1 \dot{\theta}_1^2 \sin \theta_1 - m_2 l_2 \dot{\theta}_2^2 \sin \theta_2 \\
m_2 l_1 l_2 \dot{\theta}_2^2 \sin(\theta_1 - \theta_2) \\
-m_2 l_1 l_2 \dot{\theta}_1^2 \sin(\theta_1 - \theta_2)
\end{bmatrix}
$$
*Note: This is the vector form $C\dot{q}$. The matrix $C$ is not unique, but the vector $C\dot{q}$ is.*

### Gravity Vector $G(q)$
$$ G(q) = \begin{bmatrix} 0 \\ (m_1 + m_2) g l_1 \sin \theta_1 \\ m_2 g l_2 \sin \theta_2 \end{bmatrix} $$

### Final Equation
$$
\begin{bmatrix}
M + m_1 + m_2 & (m_1 + m_2)l_1 c_1 & m_2 l_2 c_2 \\
(m_1 + m_2)l_1 c_1 & (m_1 + m_2)l_1^2 & m_2 l_1 l_2 c_{12} \\
m_2 l_2 c_2 & m_2 l_1 l_2 c_{12} & m_2 l_2^2
\end{bmatrix}
\begin{bmatrix} \ddot{x} \\ \ddot{\theta}_1 \\ \ddot{\theta}_2 \end{bmatrix}
+
\begin{bmatrix}
-(m_1 + m_2)l_1 s_1 \dot{\theta}_1^2 - m_2 l_2 s_2 \dot{\theta}_2^2 \\
m_2 l_1 l_2 s_{12} \dot{\theta}_2^2 \\
-m_2 l_1 l_2 s_{12} \dot{\theta}_1^2
\end{bmatrix}
+
\begin{bmatrix} 0 \\ (m_1 + m_2) g l_1 s_1 \\ m_2 g l_2 s_2 \end{bmatrix}
=
\begin{bmatrix} F \\ 0 \\ 0 \end{bmatrix}
$$
where $c_i = \cos \theta_i$, $s_i = \sin \theta_i$, $c_{12} = \cos(\theta_1 - \theta_2)$, $s_{12} = \sin(\theta_1 - \theta_2)$.

## 5. State Space Form
To simulate, we invert $M(q)$:
$$ \ddot{q} = M(q)^{-1} (\tau - C(q, \dot{q})\dot{q} - G(q)) $$
State vector $y = [x, \theta_1, \theta_2, \dot{x}, \dot{\theta}_1, \dot{\theta}_2]^T$.
$$ \dot{y} = [\dot{x}, \dot{\theta}_1, \dot{\theta}_2, \ddot{x}, \ddot{\theta}_1, \ddot{\theta}_2]^T $$
