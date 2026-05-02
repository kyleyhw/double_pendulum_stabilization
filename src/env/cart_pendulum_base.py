r"""
Shared base for cart-pendulum environments.

Mathematical model
==================
A cart of mass :math:`M` slides along the horizontal axis. :math:`N` rigid
massless poles of length :math:`l_i` and tip mass :math:`m_i` are connected
in series, each angle :math:`\theta_i` measured from the downward vertical.
The generalized coordinates are

.. math:: q = [x,\ \theta_1,\ \dots,\ \theta_N]^\top.

Lagrangian mechanics yields the standard manipulator form

.. math::

    M(q)\,\ddot q + C(q,\dot q) + G(q) = B u

where :math:`B = [1, 0, \dots, 0]^\top` (only the cart is actuated, so the
system is underactuated for :math:`N \ge 1`).

This module factors out the integrator, action plumbing, perturbation logic,
soft-bound terminations, and curriculum hooks. Subclasses implement
:py:meth:`_dynamics` (returning :math:`\dot s`) and :py:meth:`_state_dim` /
:py:meth:`_obs_dim`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.strategies.controls import ControlStrategy, VelocityControl
from src.strategies.rewards import ExponentialSwingUpReward, RewardStrategy


class CartPendulumBase(gym.Env, ABC):
    r"""
    Abstract cart-N-pendulum environment with curriculum, wind, impulses, and
    selectable integrator.

    Soft cart bounds
    ----------------
    Hard termination at :math:`|x| > x_{\max}` produces a discontinuity in the
    return landscape that destabilises the value function. The default here is a
    *soft* boundary: the agent is not terminated, but a quadratic penalty
    :math:`-k_{\rm bnd}\,\max(0, |x| - x_{\rm soft})^2` is added to the reward
    near the wall. Hard termination is retained at :math:`|x| > x_{\max}` (a much
    wider bound than the soft start) to guard against unbounded numerical states.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # --- subclass hooks ------------------------------------------------------

    @abstractmethod
    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        """Return :math:`\\dot s` given full state and cart force."""

    @property
    @abstractmethod
    def n_poles(self) -> int:
        """Number of pole links (1 for single, 2 for double)."""

    # --- construction --------------------------------------------------------

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        wind_std: float = 0.0,
        reset_mode: str = "down",
        control_strategy: Optional[ControlStrategy] = None,
        reward_strategy: Optional[RewardStrategy] = None,
        x_soft: float = 3.5,
        x_hard: float = 10.0,
        boundary_penalty_k: float = 0.1,
        integrator: str = "rk4",
        wind_max: float = 1.0,
    ) -> None:
        super().__init__()

        self.control_strategy = control_strategy or VelocityControl()
        self.reward_strategy = reward_strategy or ExponentialSwingUpReward()

        # Physics defaults (subclass may override).
        self.M: float = 1.0
        self.g: float = 9.81
        self.dt: float = 0.005

        self.reset_mode: str = reset_mode
        self.current_impulse: float = 0.0

        # Wind: stochastic horizontal force added to the cart at every step.
        # `wind_std` is the *current* magnitude; `wind_max` is the curriculum cap.
        self.wind_std: float = float(wind_std)
        self.wind_max: float = float(wind_max)
        self._user_wind_std: float = float(wind_std)  # remembered if user pinned a value

        # Soft / hard cart-position bounds.
        self.x_soft: float = float(x_soft)
        self.x_hard: float = float(x_hard)
        self.boundary_penalty_k: float = float(boundary_penalty_k)

        # Integrator selection: "rk4" (default) or "semi_implicit" (symplectic Euler).
        if integrator not in {"rk4", "semi_implicit"}:
            raise ValueError(f"Unknown integrator: {integrator!r}")
        self.integrator: str = integrator

        self.friction_cart: float = 0.0
        self.friction_pole: float = 0.0

        self.render_mode = render_mode
        self.action_space = self.control_strategy.get_action_space()
        self.observation_space = self._build_observation_space()
        self.state: np.ndarray = np.zeros(self._state_dim(), dtype=np.float32)

        # --- Pre-allocated work buffers (allocation reduction, candidate-3) ---
        # Sizes depend on `_state_dim()` and `n_poles` which are well-defined
        # by the time we get here (subclass set them before super().__init__).
        sd = self._state_dim()
        n = self.n_poles
        # RK4 stage buffers and the mid-state buffer. Float64 is pinned to
        # match the master baseline trajectory's arithmetic precision.
        self._k1 = np.empty(sd, dtype=np.float64)
        self._k2 = np.empty(sd, dtype=np.float64)
        self._k3 = np.empty(sd, dtype=np.float64)
        self._k4 = np.empty(sd, dtype=np.float64)
        self._mid = np.empty(sd, dtype=np.float64)
        self._new_state = np.empty(sd, dtype=np.float64)
        # Semi-implicit Euler temporaries.
        self._si_q_next = np.empty(1 + n, dtype=np.float64)
        self._si_v_next = np.empty(1 + n, dtype=np.float64)
        # Observation buffer: layout matches `_get_obs` (x, sins, coss, qdots).
        self._obs_buf = np.empty(2 + 3 * n, dtype=np.float32)
        # Cached env-params dict — refreshed only when `g` or `M` changes
        # (via `set_curriculum`). The strategies treat it read-only.
        self._env_params_cache: dict[str, Any] = {
            "dt": self.dt,
            "velocity_index": 1 + n,
            "max_force": 5000.0,
            "g": self.g,
            "M": self.M,
        }

    # --- subclass-controlled state shape -------------------------------------

    def _state_dim(self) -> int:
        # Default: q = [x, theta_1, ..., theta_N], qdot of same shape.
        return 2 * (1 + self.n_poles)

    def _build_observation_space(self) -> spaces.Box:
        # [x, sin theta_1, cos theta_1, ..., sin theta_N, cos theta_N, x_dot, theta_1_dot, ..., theta_N_dot]
        n = self.n_poles
        high = np.concatenate(
            [
                np.array([self.x_hard], dtype=np.float32),
                np.ones(2 * n, dtype=np.float32),  # sin/cos pairs
                np.full(1 + n, np.inf, dtype=np.float32),  # x_dot, theta_dot
            ]
        )
        return spaces.Box(low=-high, high=high, dtype=np.float32)

    # --- curriculum ----------------------------------------------------------

    def set_curriculum(self, difficulty: float) -> dict:
        r"""
        Apply curriculum schedule to physics, reward, and perturbations.

        Schedule
        --------
        * Gravity:    :math:`g(\delta) = 2.0 + (9.81 - 2.0)\delta`.
        * Cart friction:  :math:`\mu_{\text{cart}}(\delta) = 0.5(1-\delta)`.
        * Pole friction:  :math:`\mu_{\text{pole}}(\delta) = 0.1(1-\delta)`.
        * Wind std:  :math:`\sigma_w(\delta) = \delta\,\sigma_{w,\max}`.
        * Reward strategy receives :math:`\delta` directly.

        The user-supplied ``wind_std`` (if any) is *not* overwritten unless
        :py:meth:`set_wind_pinned` was used, ensuring inference scripts that
        explicitly set wind keep doing so.
        """
        d = float(np.clip(difficulty, 0.0, 1.0))

        self.g = 2.0 + d * (9.81 - 2.0)
        self.friction_cart = 0.5 * (1.0 - d)
        self.friction_pole = 0.1 * (1.0 - d)

        if self._user_wind_std == 0.0:
            self.wind_std = d * self.wind_max

        self.reward_strategy.set_curriculum(d)
        # Refresh env-params cache so the next step sees the new gravity.
        # The cache is otherwise stable across steps; only g (and rarely M)
        # changes during curriculum advancement.
        self._env_params_cache["g"] = self.g
        self._env_params_cache["M"] = self.M
        return {
            "g": self.g,
            "friction_cart": self.friction_cart,
            "friction_pole": self.friction_pole,
            "wind_std": self.wind_std,
            "reward_difficulty": d,
        }

    def set_wind_pinned(self, wind_std: float) -> None:
        """Pin wind to a user-supplied value (curriculum no longer overrides)."""
        self._user_wind_std = float(wind_std)
        self.wind_std = float(wind_std)

    # --- reset / step --------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Random small perturbation around the base configuration.
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self._state_dim(),))

        mode = self.reset_mode
        if options and "mode" in options:
            mode = options["mode"]

        # Angle indices in the full state are 1 .. n_poles inclusive.
        if mode == "up":
            for i in range(1, 1 + self.n_poles):
                self.state[i] += np.pi
        elif mode == "down":
            pass
        elif mode == "random":
            for i in range(1, 1 + self.n_poles):
                self.state[i] = self.np_random.uniform(0.0, 2 * np.pi)
        else:
            raise ValueError(f"Unknown reset mode: {mode!r}")

        self.reward_strategy.reset()
        self.current_impulse = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # 1. Force = control strategy applied to action (and current state).
        # Reuse the cached env_params dict — refreshed only on curriculum
        # changes — instead of allocating a fresh dict per step.
        env_params = self._env_params_cache
        force = float(self.control_strategy.get_force(action, self.state, env_params))

        # 2. Wind + impulse perturbations.
        if self.wind_std > 0.0:
            force += float(self.np_random.normal(0.0, self.wind_std))
        force += self.current_impulse
        self.current_impulse = 0.0

        # 3. Integrate.
        self.state = self._integrate(self.state, force, self.dt)

        x = float(self.state[0])

        # 4. Boundary handling.
        terminated = bool(abs(x) > self.x_hard)

        # 5. Reward = strategy's per-step reward + soft-boundary penalty.
        reward = float(self.reward_strategy.compute_reward(self.state, env_params))
        if abs(x) > self.x_soft:
            overshoot = abs(x) - self.x_soft
            reward -= self.boundary_penalty_k * overshoot * overshoot

        return self._get_obs(), reward, terminated, False, {}

    # --- integrators ---------------------------------------------------------

    def _integrate(self, state: np.ndarray, force: float, dt: float) -> np.ndarray:
        if self.integrator == "rk4":
            return self._rk4_step(state, force, dt)
        return self._semi_implicit_step(state, force, dt)

    def _rk4_step(self, state: np.ndarray, force: float, dt: float) -> np.ndarray:
        r"""
        Classical 4th-order Runge-Kutta. Not symplectic; energy drifts as
        :math:`\mathcal O(dt^4)` per step but is *not* conserved on long
        chaotic trajectories.

        Buffer-reusing implementation
        -----------------------------
        Subclasses may opt in to in-place dynamics by implementing
        :py:meth:`_dynamics_into(state, force, out)`. When present, RK4
        writes derivatives directly into pre-allocated stage buffers
        (:py:attr:`_k1`..`_k4`) and accumulates the update in
        :py:attr:`_new_state`, avoiding the four allocations the legacy
        path made per step. The arithmetic is preserved verbatim
        (state + 0.5*dt*k_i, then state + (dt/6)(k1 + 2 k2 + 2 k3 + k4))
        so the float64 trajectory is bit-identical to the legacy path.

        Subclasses without `_dynamics_into` fall back to the original
        allocate-each-stage form for backwards compatibility.
        """
        if hasattr(self, "_dynamics_into"):
            return self._rk4_step_inplace(state, force, dt)
        # Legacy path (unmodified arithmetic).
        k1 = self._dynamics(state, force)
        k2 = self._dynamics(state + 0.5 * dt * k1, force)
        k3 = self._dynamics(state + 0.5 * dt * k2, force)
        k4 = self._dynamics(state + dt * k3, force)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_step_inplace(self, state: np.ndarray, force: float, dt: float) -> np.ndarray:
        """In-place RK4 stage; requires `_dynamics_into`. See `_rk4_step`."""
        k1, k2, k3, k4 = self._k1, self._k2, self._k3, self._k4
        mid = self._mid

        # Stage 1: k1 = f(state)
        self._dynamics_into(state, force, k1)

        # Stage 2: mid = state + 0.5*dt*k1; k2 = f(mid)
        # Order of ops matches `state + 0.5 * dt * k1`: numpy evaluates
        # `0.5 * dt * k1` left-to-right, so `(0.5*dt) * k1` then `state + ...`.
        np.multiply(k1, 0.5 * dt, out=mid)
        np.add(mid, state, out=mid)
        self._dynamics_into(mid, force, k2)

        # Stage 3
        np.multiply(k2, 0.5 * dt, out=mid)
        np.add(mid, state, out=mid)
        self._dynamics_into(mid, force, k3)

        # Stage 4
        np.multiply(k3, dt, out=mid)
        np.add(mid, state, out=mid)
        self._dynamics_into(mid, force, k4)

        # Combine: new = state + (dt/6) * (k1 + 2 k2 + 2 k3 + k4)
        # Mirror original parenthesisation so float64 rounding is identical.
        ns = self._new_state
        np.multiply(k2, 2.0, out=mid)             # mid = 2 k2
        np.add(k1, mid, out=mid)                  # mid = k1 + 2 k2
        np.multiply(k3, 2.0, out=ns)              # ns  = 2 k3
        np.add(mid, ns, out=mid)                  # mid = k1 + 2 k2 + 2 k3
        np.add(mid, k4, out=mid)                  # mid = k1 + 2 k2 + 2 k3 + k4
        np.multiply(mid, dt / 6.0, out=ns)
        np.add(ns, state, out=ns)
        return ns.copy()

    def _semi_implicit_step(self, state: np.ndarray, force: float, dt: float) -> np.ndarray:
        r"""
        Semi-implicit (symplectic) Euler.

        Update rule (with :math:`q = [x, \theta_1, \dots, \theta_N]`,
        :math:`v = \dot q`):

        .. math::

            v_{t+1} = v_t + dt \cdot \ddot q(q_t, v_t, F_t),
            \qquad q_{t+1} = q_t + dt \cdot v_{t+1}.

        First-order accurate but exactly conserves a *modified* Hamiltonian,
        bounding energy drift on long horizons. Faster per step than RK4 (1
        dynamics call) but less accurate locally; useful when long-horizon
        physics fidelity matters more than per-step error.

        Buffer-reusing implementation: the two slices `q_next` and `v_next`
        are written into pre-allocated buffers and concatenated into
        `_new_state`. The arithmetic (a = f[n:], v_next = v + dt*a,
        q_next = q + dt*v_next, concatenate) is unchanged.
        """
        n = self._state_dim() // 2
        if hasattr(self, "_dynamics_into"):
            f = self._k1  # reuse k1 as the dynamics output buffer
            self._dynamics_into(state, force, f)
        else:
            f = self._dynamics(state, force)

        # Slice views — copies are still required because state is the
        # *current* state and we write to internal buffers that may alias.
        q = state[:n]
        v = state[n:]
        a = f[n:]
        # v_next = v + dt * a  (compute into _si_v_next)
        np.multiply(a, dt, out=self._si_v_next)
        np.add(self._si_v_next, v, out=self._si_v_next)
        # q_next = q + dt * v_next
        np.multiply(self._si_v_next, dt, out=self._si_q_next)
        np.add(self._si_q_next, q, out=self._si_q_next)
        # Concatenate (single allocation, replacing the original
        # np.concatenate that allocated a 2N-array each call).
        ns = self._new_state
        ns[:n] = self._si_q_next
        ns[n:] = self._si_v_next
        return ns.copy()

    # --- observation / utility ----------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # Buffer-reusing form: write trig-encoded obs into the pre-allocated
        # float32 buffer and return a *copy*. The copy preserves the public
        # contract that the returned array is independent of subsequent
        # internal mutation (test_observation_constructor_idempotent in
        # tests/test_pipeline_equivalence.py asserts this).
        n = self.n_poles
        s = self.state
        buf = self._obs_buf
        buf[0] = s[0]
        # sins (positions 1..n) and coss (positions n+1..2n).
        # Use ufunc with out= to write directly into the buffer; np.sin/np.cos
        # over a 1-2 element slice has identical float32 result to assigning
        # element-wise.
        thetas = s[1 : 1 + n]
        np.sin(thetas, out=buf[1 : 1 + n])
        np.cos(thetas, out=buf[1 + n : 1 + 2 * n])
        # qdots
        buf[1 + 2 * n : 2 + 3 * n] = s[1 + n :]
        return buf.copy()

    def _env_params(self) -> dict[str, Any]:
        """Bundle of physics parameters consumed by control and reward strategies.

        Backwards-compat wrapper that returns the cached dict (used internally
        by `step`). External callers should treat the returned dict as
        read-only.
        """
        return self._env_params_cache

    def apply_impulse(self, force: float) -> None:
        """Apply a single-step impulsive force (added to the next ``step``)."""
        self.current_impulse = float(force)

    def render(self) -> None:  # pragma: no cover - rendering done by Visualizer
        return None
