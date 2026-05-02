r"""
Reward strategies.

The strategy interface :class:`RewardStrategy` exposes:

* :py:meth:`compute_reward` — instantaneous scalar reward :math:`r_t`.
* :py:meth:`set_curriculum` — adjust internal parameters with the difficulty
  knob :math:`\delta \in [0, 1]`.
* :py:meth:`reset` — clear per-episode state (continuity counters, etc.).

All concrete strategies in this module work for both the single- and
double-pendulum envs by inspecting the state vector length.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RewardStrategy(ABC):
    @abstractmethod
    def compute_reward(self, state: np.ndarray, env_params: dict[str, Any]) -> float: ...

    @abstractmethod
    def set_curriculum(self, difficulty: float) -> None: ...

    def reset(self) -> None:
        """Clear per-episode internal state. Default: no-op."""
        return None


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def _angles_from_state(state: np.ndarray) -> tuple[float, list[float]]:
    n = len(state)
    if n == 6:
        return float(state[0]), [float(state[1]), float(state[2])]
    if n == 4:
        return float(state[0]), [float(state[1])]
    n_pends = (n // 2) - 1
    return float(state[0]), list(map(float, state[1 : 1 + n_pends]))


def _angle_to_up_error(theta: float) -> float:
    r"""Shortest-magnitude distance to the upright equilibrium :math:`\theta = \pi`."""
    return float(np.abs(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))))


def _curriculum_threshold(difficulty: float, curve: str = "linear") -> float:
    r"""
    Threshold schedule :math:`\epsilon(\delta)`. Two shapes are supported.

    * ``"linear"`` (default; matches the original design):
      :math:`\epsilon(\delta) = 90^\circ - 80^\circ \delta`.

    * ``"concave"``: :math:`\epsilon(\delta) = 10^\circ + 80^\circ \sqrt{1 - \delta}`.
      This stays *wider* at every intermediate :math:`\delta` than the linear
      schedule (e.g. :math:`+10^\circ` at :math:`\delta = 0.27`), giving the
      agent more tolerance during the difficulty regime where it is still
      learning to balance, before the band tightens sharply near :math:`\delta = 1`.
    """
    min_a = np.deg2rad(10.0)
    max_a = np.deg2rad(90.0)
    d = float(difficulty)
    if curve == "linear":
        return max_a - (max_a - min_a) * d
    if curve == "concave":
        return min_a + (max_a - min_a) * float(np.sqrt(max(0.0, 1.0 - d)))
    raise ValueError(f"Unknown threshold curve: {curve!r}. Use 'linear' or 'concave'.")


def _potential_energy(state: np.ndarray, l1: float = 1.0, l2: float = 1.0,
                      m1: float = 0.5, m2: float = 0.5, g: float = 9.81) -> float:
    """V(q) for one or two pendulums (cart contributes nothing)."""
    n = len(state)
    if n == 4:
        theta = state[1]
        return float(-m1 * g * l1 * np.cos(theta))
    if n == 6:
        t1, t2 = state[1], state[2]
        return float(-(m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2))
    raise ValueError(f"Unsupported state dim: {n}")


def _kinetic_energy(state: np.ndarray, M: float = 1.0, l1: float = 1.0, l2: float = 1.0,
                    m1: float = 0.5, m2: float = 0.5) -> float:
    """T(q, qdot) using the manipulator mass matrix."""
    n = len(state)
    if n == 4:
        x_dot, t_dot = state[2], state[3]
        c = np.cos(state[1])
        return float(0.5 * ((M + m1) * x_dot ** 2 + m1 * l1 ** 2 * t_dot ** 2
                            + 2.0 * m1 * l1 * c * x_dot * t_dot))
    if n == 6:
        t1, t2 = state[1], state[2]
        x_dot, t1d, t2d = state[3], state[4], state[5]
        c1, c2 = np.cos(t1), np.cos(t2)
        c12 = np.cos(t1 - t2)
        return float(
            0.5 * (
                (M + m1 + m2) * x_dot ** 2
                + (m1 + m2) * l1 ** 2 * t1d ** 2
                + m2 * l2 ** 2 * t2d ** 2
                + 2.0 * (m1 + m2) * l1 * c1 * x_dot * t1d
                + 2.0 * m2 * l2 * c2 * x_dot * t2d
                + 2.0 * m2 * l1 * l2 * c12 * t1d * t2d
            )
        )
    raise ValueError(f"Unsupported state dim: {n}")


# ---------------------------------------------------------------------------- #
# Sparse "all-up" rewards
# ---------------------------------------------------------------------------- #

class _ThresholdCurriculumMixin:
    """Holds the threshold-curve choice. Subclasses set ``self.threshold_curve`` in __init__."""
    threshold_curve: str = "linear"


class DoublePendulumStandardReward(RewardStrategy):
    r"""
    Sparse all-up reward with optional smooth survival bonus.

    .. math::

        r_t = \begin{cases}
            1 + 0.5\,\mathbb 1(|x| < 1) & \text{both poles within } \epsilon \text{ of } \pi \\
            \alpha\,\cos\theta_1\,\cos\theta_2 + 0.1(1 - \delta) & \text{otherwise}
        \end{cases}

    The :math:`\alpha\,\cos\theta_1\,\cos\theta_2` term (default :math:`\alpha = 0`)
    is a smooth survival bonus: :math:`-1` when both poles point straight down,
    :math:`+1` when both point up. It provides a non-trivial gradient outside the
    threshold band but does not affect the optimum (which still places both
    poles at :math:`\pi`).
    """
    def __init__(self, survival_alpha: float = 0.0, threshold_curve: str = "linear") -> None:
        self.reward_threshold: float = float(np.deg2rad(10.0))
        self.difficulty: float = 1.0
        self.survival_alpha: float = float(survival_alpha)
        self.threshold_curve: str = threshold_curve

    def set_curriculum(self, difficulty: float) -> None:
        self.difficulty = float(difficulty)
        self.reward_threshold = _curriculum_threshold(difficulty, self.threshold_curve)

    def compute_reward(self, state: np.ndarray, env_params: dict[str, Any]) -> float:
        x, angles = _angles_from_state(state)
        if len(angles) < 2:
            raise ValueError("DoublePendulumStandardReward requires 2 poles.")
        d1 = _angle_to_up_error(angles[0])
        d2 = _angle_to_up_error(angles[1])
        if d1 < self.reward_threshold and d2 < self.reward_threshold:
            return 1.0 + (0.5 if abs(x) < 1.0 else 0.0)
        survival = self.survival_alpha * float(np.cos(angles[0]) * np.cos(angles[1]))
        return survival + 0.1 * (1.0 - self.difficulty)


class SinglePendulumStandardReward(RewardStrategy):
    """Sparse upright reward for the single pendulum."""
    def __init__(self, survival_alpha: float = 0.0, threshold_curve: str = "linear") -> None:
        self.reward_threshold: float = float(np.deg2rad(10.0))
        self.difficulty: float = 1.0
        self.survival_alpha: float = float(survival_alpha)
        self.threshold_curve: str = threshold_curve

    def set_curriculum(self, difficulty: float) -> None:
        self.difficulty = float(difficulty)
        self.reward_threshold = _curriculum_threshold(difficulty, self.threshold_curve)

    def compute_reward(self, state: np.ndarray, env_params: dict[str, Any]) -> float:
        x, angles = _angles_from_state(state)
        d1 = _angle_to_up_error(angles[0])
        if d1 < self.reward_threshold:
            return 1.0 + (0.5 if abs(x) < 1.0 else 0.0)
        # cos(theta) = -1 down, +1 up — same shaping as double-pendulum survival.
        survival = self.survival_alpha * float(np.cos(angles[0]))
        return survival + 0.1 * (1.0 - self.difficulty)


# ---------------------------------------------------------------------------- #
# Exponential continuity reward
# ---------------------------------------------------------------------------- #

class ExponentialSwingUpReward(RewardStrategy):
    r"""
    Exponential-continuity reward.

    Let :math:`S_t` be the all-up indicator and :math:`T_{up,t}` the duration
    (in seconds) of the current uninterrupted upright run. Define

    .. math::

        r_t = \mathbb 1(S_t) \cdot \bigl(\exp(\min(T_{up,t}, T_{cap})) - 1\bigr)
              \cdot P_x(x).

    The position term :math:`P_x` is **gated**: at low difficulty it is
    effectively flat (so the agent is free to swing the cart for energy
    pumping); at high difficulty it tightens to enforce centring. Specifically

    .. math::

        \sigma_x(\delta) = \sigma_{x,\max} - (\sigma_{x,\max} - \sigma_{x,\min})\,\delta
        ,\quad
        P_x(x) = \exp\!\bigl(-x^2 / (2\sigma_x^2(\delta))\bigr).

    With defaults :math:`\sigma_{x,\max} = 5` and :math:`\sigma_{x,\min} = 1.5`
    the swing-up phase pays essentially no centring cost (:math:`P_x \approx 1`
    over :math:`|x| < 4`), while the late-stage stabilisation penalises drift.
    """
    DEFAULT_T_CAP: float = 5.0
    DEFAULT_SIGMA_X_MAX: float = 5.0
    DEFAULT_SIGMA_X_MIN: float = 1.5

    def __init__(self, t_cap: float = DEFAULT_T_CAP,
                 sigma_x_max: float = DEFAULT_SIGMA_X_MAX,
                 sigma_x_min: float = DEFAULT_SIGMA_X_MIN,
                 threshold_curve: str = "linear") -> None:
        self.reward_threshold: float = float(np.deg2rad(10.0))
        self.difficulty: float = 1.0
        self.steps_above_threshold: int = 0
        self.t_cap: float = float(t_cap)
        self.sigma_x_max: float = float(sigma_x_max)
        self.sigma_x_min: float = float(sigma_x_min)
        self.threshold_curve: str = threshold_curve

    def reset(self) -> None:
        self.steps_above_threshold = 0

    def set_curriculum(self, difficulty: float) -> None:
        self.difficulty = float(difficulty)
        self.reward_threshold = _curriculum_threshold(difficulty, self.threshold_curve)

    def _sigma_x(self) -> float:
        return self.sigma_x_max - (self.sigma_x_max - self.sigma_x_min) * self.difficulty

    def compute_reward(self, state: np.ndarray, env_params: dict[str, Any]) -> float:
        dt = float(env_params.get("dt", 0.005))
        x, angles = _angles_from_state(state)

        all_up = all(_angle_to_up_error(t) < self.reward_threshold for t in angles)
        if not all_up:
            self.steps_above_threshold = 0
            return 0.0

        self.steps_above_threshold += 1
        time_above = min(self.steps_above_threshold * dt, self.t_cap)
        magnitude = float(np.exp(time_above) - 1.0)
        sigma_x = self._sigma_x()
        pos = float(np.exp(-(x ** 2) / (2.0 * sigma_x ** 2)))
        return magnitude * pos

    def max_per_step_reward(self) -> float:
        return float(np.exp(self.t_cap) - 1.0)


# ---------------------------------------------------------------------------- #
# Energy-shaping reward (Tedrake-style swing-up)
# ---------------------------------------------------------------------------- #

class EnergyShapingReward(RewardStrategy):
    r"""
    Energy-shaping reward.

    For a pendulum on a cart, the energy at the upright equilibrium with zero
    velocity is :math:`E^* = +(\sum_i m_i) g l` (potential maximum, no kinetic).
    The energy error

    .. math:: \Delta E(s) = E(s) - E^*

    is a global signal: its zero-set is exactly the homoclinic orbit through
    upright. We reward proximity to this manifold AND alignment with upright,
    via the convex combination

    .. math::

        r_t = w_E \exp\!\Bigl(-\frac{\Delta E^2}{2 \sigma_E^2}\Bigr)
            + w_S \exp\!\Bigl(-\frac{\sum_i e_{\theta_i}^2}{2 \sigma_\theta^2(\delta)}\Bigr)
            + w_K \exp\!\Bigl(-\frac{T(s)}{\tau_K}\Bigr)\,\mathbb 1(\sum_i e_{\theta_i}^2 < \sigma_\theta^2(\delta)).

    Components
    ----------
    * **Energy** :math:`w_E`: pulls the agent onto the homoclinic manifold
      (provides gradient even when the pendulum is far from upright).
    * **Spatial** :math:`w_S`: aligns the angles with :math:`\pi` once the
      energy is right.
    * **Kinetic damping** :math:`w_K`: gated on being near upright; rewards
      stopping (:math:`T \to 0`).

    The spatial scale :math:`\sigma_\theta` shrinks with curriculum
    :math:`\delta`, mirroring the threshold curriculum used by the sparse
    rewards.
    """
    def __init__(
        self,
        w_energy: float = 0.4,
        w_spatial: float = 0.4,
        w_kinetic: float = 0.2,
        sigma_energy: float = 5.0,
        sigma_theta_max: float = np.deg2rad(90.0),
        sigma_theta_min: float = np.deg2rad(10.0),
        kinetic_tau: float = 5.0,
        threshold_curve: str = "linear",
    ) -> None:
        self.w_energy = float(w_energy)
        self.w_spatial = float(w_spatial)
        self.w_kinetic = float(w_kinetic)
        self.sigma_energy = float(sigma_energy)
        self.sigma_theta_max = float(sigma_theta_max)
        self.sigma_theta_min = float(sigma_theta_min)
        self.kinetic_tau = float(kinetic_tau)
        self.threshold_curve: str = threshold_curve
        self.difficulty: float = 1.0
        # Threshold attribute kept for compatibility with the trainer's CSV log.
        self.reward_threshold: float = self.sigma_theta_max

    def reset(self) -> None:
        return None

    def set_curriculum(self, difficulty: float) -> None:
        self.difficulty = float(difficulty)
        self.reward_threshold = self._sigma_theta()

    def _sigma_theta(self) -> float:
        d = float(self.difficulty)
        if self.threshold_curve == "linear":
            return self.sigma_theta_max - (self.sigma_theta_max - self.sigma_theta_min) * d
        if self.threshold_curve == "concave":
            return self.sigma_theta_min + (self.sigma_theta_max - self.sigma_theta_min) * float(np.sqrt(max(0.0, 1.0 - d)))
        raise ValueError(f"Unknown threshold curve: {self.threshold_curve!r}")

    def _e_target(self, state: np.ndarray, env_params: dict[str, Any]) -> tuple[float, float]:
        """Return (E_current, E_target). Uses default body parameters; OK because the
        env passes `g` indirectly via the dynamics, but here we approximate using
        the standard l = 1, m = 0.5 used by the envs.
        """
        n = len(state)
        # Pull g from env_params if available (we plumb it through env_params['g']).
        g = float(env_params.get("g", 9.81))
        if n == 4:
            E_target = +0.5 * g * 1.0  # +m g l (single pendulum at upright, m=0.5, l=1)
            T = _kinetic_energy(state, M=1.0, l1=1.0, m1=0.5)
            V = _potential_energy(state, l1=1.0, m1=0.5, g=g)
        elif n == 6:
            E_target = +(0.5 + 0.5) * g * 1.0 + 0.5 * g * 1.0  # (m1+m2) g l1 + m2 g l2
            T = _kinetic_energy(state, M=1.0, l1=1.0, l2=1.0, m1=0.5, m2=0.5)
            V = _potential_energy(state, l1=1.0, l2=1.0, m1=0.5, m2=0.5, g=g)
        else:
            raise ValueError(f"Unsupported state dim: {n}")
        return T + V, E_target

    def compute_reward(self, state: np.ndarray, env_params: dict[str, Any]) -> float:
        x, angles = _angles_from_state(state)
        E, E_target = self._e_target(state, env_params)
        dE = E - E_target

        sigma_t = max(1e-3, self._sigma_theta())
        angle_err_sq = float(sum(_angle_to_up_error(t) ** 2 for t in angles))

        r_energy = float(np.exp(-(dE ** 2) / (2.0 * self.sigma_energy ** 2)))
        r_spatial = float(np.exp(-angle_err_sq / (2.0 * sigma_t ** 2)))

        T_kin = _kinetic_energy(state, M=1.0, l1=1.0, l2=1.0, m1=0.5, m2=0.5) \
            if len(state) == 6 else _kinetic_energy(state, M=1.0, l1=1.0, m1=0.5)
        gating = float(angle_err_sq < sigma_t ** 2)
        r_kinetic = float(np.exp(-max(T_kin, 0.0) / self.kinetic_tau)) * gating

        return self.w_energy * r_energy + self.w_spatial * r_spatial + self.w_kinetic * r_kinetic

    def max_per_step_reward(self) -> float:
        return float(self.w_energy + self.w_spatial + self.w_kinetic)


# ---------------------------------------------------------------------------- #
# LQR-style quadratic-cost reward (DENSE, smooth, all-the-way-to-optimum)
# ---------------------------------------------------------------------------- #

class LQRCostReward(RewardStrategy):
    r"""
    LQR-style quadratic-cost reward.

    Why
    ---
    Sparse / step-function rewards (``ExponentialSwingUpReward``,
    ``*StandardReward``) provide *no* gradient inside the threshold band, so PPO
    has no signal pulling the angles tighter than the band edge. Empirically
    this caps strict-success rates at ~3% of episode time regardless of how
    long we train. A quadratic cost on the state has gradient

    .. math:: \nabla_s (-s^T Q s) = -2 Q s

    everywhere, including arbitrarily close to upright -- the policy gradient
    pulls toward zero state error all the way to the optimum.

    Mathematical form
    -----------------
    With state error :math:`\tilde s = s - s^*` (where :math:`s^*` is the
    upright reference, :math:`\theta_i = \pi`), the per-step reward is

    .. math:: r_t = -\tilde s^T Q \tilde s - R \, a_t^2 + b_{\rm alive},

    expanded as

    .. math::

        r_t = b_{\rm alive}
            - q_x x^2
            - q_\theta \sum_i e_{\theta_i}^2
            - q_{\dot x} \dot x^2
            - q_{\dot\theta} \sum_i \dot\theta_i^2
            - R \, a_t^2.

    The angle error :math:`e_{\theta_i}` uses the wrap-around metric to
    upright, so :math:`e_{\theta_i} \in [0, \pi]` regardless of how the agent
    arrives.

    Curriculum
    ----------
    The angle weight :math:`q_\theta(\delta)` schedules from a low value at
    :math:`\delta = 0` (so the swing-up phase is dominated by velocity damping
    + position centring, not by demanding immediate angle alignment) to a
    high value at :math:`\delta = 1` (tight stabilisation):

    .. math:: q_\theta(\delta) = q_\theta^{\min} + (q_\theta^{\max} - q_\theta^{\min}) \, \delta.

    The schedule is linear in :math:`\delta`. Unlike the threshold curriculum
    in the sparse rewards, here :math:`\delta` modulates *gradient strength*
    rather than a hard band, so there is no discontinuity to fall off.

    Reward magnitude is bounded above by :math:`b_{\rm alive}` (when
    :math:`s = s^*`) and unbounded below — typical episode reward will be
    negative early in training and approach :math:`b_{\rm alive} \cdot T`
    at convergence.
    """
    def __init__(
        self,
        q_x: float = 0.1,
        q_theta_min: float = 2.0,
        q_theta_max: float = 5.0,
        q_xdot: float = 0.01,
        q_thetadot: float = 0.05,
        r_action: float = 0.005,
        b_alive: float = 1.0,
        # Velocity penalties are gated by proximity to upright using a
        # Gaussian envelope of width sigma_proximity (radians). Far from
        # upright (during swing-up) the velocity terms are effectively zero;
        # near upright they dominate and force damping. This breaks the
        # "do nothing is optimal at down-rest" attractor that ungated
        # quadratic velocity penalties produce. Empirically this was the
        # failure mode of the first LQR-reward attempt (Phase H).
        sigma_proximity: float = np.deg2rad(45.0),
        # Threshold attribute is kept for trainer-CSV compatibility but plays
        # no role in this reward's gradient.
        nominal_threshold_deg: float = 10.0,
    ) -> None:
        self.q_x = float(q_x)
        self.q_theta_min = float(q_theta_min)
        self.q_theta_max = float(q_theta_max)
        self.q_xdot = float(q_xdot)
        self.q_thetadot = float(q_thetadot)
        self.r_action = float(r_action)
        self.b_alive = float(b_alive)
        self.sigma_proximity = float(sigma_proximity)
        self.difficulty: float = 1.0
        self.q_theta: float = self.q_theta_max
        # For CSV / trainer compatibility (the trainer reads this field).
        self.reward_threshold: float = float(np.deg2rad(nominal_threshold_deg))
        # Last action seen (the env passes the *force*, not the normalised
        # action, so we retain a lightweight estimate of its magnitude
        # via the state's velocity change; see compute_reward).
        self._last_action: float = 0.0

    def reset(self) -> None:
        self._last_action = 0.0

    def set_curriculum(self, difficulty: float) -> None:
        self.difficulty = float(difficulty)
        self.q_theta = self.q_theta_min + (self.q_theta_max - self.q_theta_min) * self.difficulty

    def compute_reward(self, state: np.ndarray, env_params: dict[str, Any]) -> float:
        x, angles = _angles_from_state(state)
        n_poles = len(angles)
        x_dot = float(state[1 + n_poles])
        theta_dots = [float(state[1 + n_poles + 1 + i]) for i in range(n_poles)]

        angle_err_sq = float(sum(_angle_to_up_error(t) ** 2 for t in angles))
        velocity_sq = float(sum(td ** 2 for td in theta_dots))

        # Proximity-to-upright envelope: Gaussian of the angle error.
        # Equals 1 at perfect upright and decays smoothly away.
        # Velocity penalties are scaled by this so they don't punish
        # large swings during swing-up but still force stillness near upright.
        proximity = float(np.exp(-angle_err_sq / (2.0 * self.sigma_proximity ** 2)))

        cost = (
            self.q_x * (x * x)
            + self.q_theta * angle_err_sq
            + proximity * self.q_xdot * (x_dot * x_dot)
            + proximity * self.q_thetadot * velocity_sq
        )

        # Action quadratic penalty. Uses the optional last-action hook;
        # zero by default so the reward is independent of the action vector.
        action_pen = self.r_action * (self._last_action ** 2)

        # b_alive ensures that the reward is positive at the optimum (s = s*,
        # u = 0), which is helpful for the ratchet's "beat-the-best" gate
        # and for keeping cumulative episode reward interpretable.
        return self.b_alive - cost - action_pen

    def observe_action(self, action_norm: float) -> None:
        """
        Optional hook: the env or trainer can call this to inform the next
        compute_reward of the action that was just applied. Unused for the
        out-of-the-box pipeline (we measure cost from the state alone), but
        retained for completeness.
        """
        self._last_action = float(action_norm)

    def max_per_step_reward(self) -> float:
        # Best case: s = s*, u = 0 -> r = b_alive.
        return float(self.b_alive)


# ---------------------------------------------------------------------------- #
# Hybrid: exponential continuity (swing-up) + LQR quadratic (tight stab)
# ---------------------------------------------------------------------------- #

class HybridLQRSwingUpReward(RewardStrategy):
    r"""
    Hybrid reward combining ``ExponentialSwingUpReward`` (proven swing-up
    gradient, gives a large band-entry bonus that scales with continuity) with
    a *small* LQR-style quadratic angle penalty (always-on, gives a gradient
    inside the band that pulls the angle toward exactly :math:`\pi`).

    .. math::

        r_t = r_{\rm exp}(s; \delta) - q_\theta(\delta) \sum_i e_{\theta_i}^2 - q_x x^2.

    Why this composition
    --------------------
    Phases B/E/F/G showed that the exponential reward drives swing-up reliably
    and reaches :math:`\delta \approx 0.30`, but its strict-success rate stalls
    at ~3 % because :math:`\partial r_{\rm exp}/\partial \theta = 0` *inside*
    the threshold band — the policy has no gradient to tighten further.
    Phases H/H2 showed that a pure LQR quadratic cost has no swing-up gradient
    and the agent never reaches upright at all.

    Adding a small quadratic angle penalty to the exponential reward gives the
    best of both:
      - **Outside the band**: the quadratic penalty is the dominant signal
        (the exp reward is zero), but its magnitude is bounded
        (:math:`q_\theta \cdot 2\pi^2 \approx 4` per step at default
        :math:`q_\theta = 0.2`), so it does not overwhelm the per-step exp
        reward of :math:`\sim 100`–:math:`147` once the agent reaches
        upright.
      - **Inside the band**: the exp reward dominates in magnitude, but the
        quadratic penalty supplies a non-zero gradient toward
        :math:`\theta = \pi`. This directly addresses the strict-success
        ceiling identified in the cross-phase analysis.

    The :math:`q_x x^2` term mirrors the cart-position penalty already present
    in ``ExponentialSwingUpReward``'s :math:`P_x(x)` term, but with consistent
    quadratic gradient (not Gaussian-flat at large :math:`|x|`).
    """
    def __init__(
        self,
        exp_t_cap: float = ExponentialSwingUpReward.DEFAULT_T_CAP,
        sigma_x_max: float = ExponentialSwingUpReward.DEFAULT_SIGMA_X_MAX,
        sigma_x_min: float = ExponentialSwingUpReward.DEFAULT_SIGMA_X_MIN,
        threshold_curve: str = "linear",
        q_theta: float = 0.2,
        q_x: float = 0.05,
    ) -> None:
        self._exp = ExponentialSwingUpReward(
            t_cap=exp_t_cap, sigma_x_max=sigma_x_max, sigma_x_min=sigma_x_min,
            threshold_curve=threshold_curve,
        )
        self.q_theta = float(q_theta)
        self.q_x = float(q_x)
        # Mirror the threshold attribute so the trainer's CSV log works.
        self.reward_threshold: float = self._exp.reward_threshold
        self.threshold_curve = threshold_curve
        self.difficulty: float = 1.0

    def reset(self) -> None:
        self._exp.reset()

    def set_curriculum(self, difficulty: float) -> None:
        self.difficulty = float(difficulty)
        self._exp.set_curriculum(difficulty)
        self.reward_threshold = self._exp.reward_threshold

    def compute_reward(self, state: np.ndarray, env_params: dict[str, Any]) -> float:
        x, angles = _angles_from_state(state)
        r_exp = self._exp.compute_reward(state, env_params)
        angle_err_sq = float(sum(_angle_to_up_error(t) ** 2 for t in angles))
        return r_exp - self.q_theta * angle_err_sq - self.q_x * x * x

    def max_per_step_reward(self) -> float:
        return self._exp.max_per_step_reward()
