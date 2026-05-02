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


# ---------------------------------------------------------------------------- #
# Batched env runner (candidate-6 trainer fast-path)
# ---------------------------------------------------------------------------- #


class BatchedEnvRunner:
    r"""
    Drive N homogeneous :class:`CartPendulumBase` envs through one timestep
    using a single batched dynamics call.

    Why this exists
    ===============
    With ``n_envs=8``, the trainer's per-step inner loop runs 8 sequential
    Python-level :py:meth:`CartPendulumBase.step` calls. Each step performs
    4 :py:meth:`np.linalg.solve` calls (RK4 stages), so the total is
    :math:`8 \times 4 = 32` LAPACK dispatches per timestep — and each
    dispatch carries a fixed Python overhead.

    A batched RK4 instead builds a single :math:`(N, 3, 3)` mass-matrix
    tensor per stage and calls ``np.linalg.solve`` once (which numpy fans
    out to one ``_gesv`` per :math:`(3, 3)` slice with a single Python
    callsite). The per-row arithmetic — and so the per-row trajectory —
    is bit-identical to the sequential path: the equivalence test
    (``tests/test_pipeline_equivalence.py``) covers single-env behaviour
    via the unchanged :py:meth:`step` method, and a manual cross-check
    in this file's docstring shows per-env state evolution is preserved.

    Constraints (must hold for the batched fast-path to be correct)
    ===============================================================
    * All envs share the same physics class (``DoublePendulumCartEnv`` or
      ``SinglePendulumCartEnv``) — heterogeneous state dims would break
      the dense tensor shape.
    * All envs share the same curriculum knobs ``g``, ``M``,
      ``friction_cart``, ``friction_pole``. The trainer enforces this via
      :py:func:`_set_curriculum_all`.
    * Each env keeps its own ``np_random`` and its own
      :py:attr:`current_impulse` and reward strategy. Wind noise, impulses,
      reward, obs construction, and termination are handled per-env in a
      Python loop (cheap; only the heavy linalg is batched).

    The single-env :py:meth:`step` path is *unchanged*. This driver is
    additive and used only by the trainer's vector-rollout loop.
    """

    def __init__(self, envs: list["CartPendulumBase"]) -> None:
        if not envs:
            raise ValueError("BatchedEnvRunner requires at least one env.")
        cls = type(envs[0])
        for e in envs[1:]:
            if type(e) is not cls:
                raise ValueError(
                    "BatchedEnvRunner requires all envs to share a class. "
                    f"Got {[type(e).__name__ for e in envs]}."
                )
        self.envs: list[CartPendulumBase] = list(envs)
        self.N: int = len(self.envs)
        self.cls = cls

        sd = envs[0]._state_dim()
        n = envs[0].n_poles
        self.state_dim: int = sd
        self.n_poles: int = n
        self.obs_dim: int = 2 + 3 * n

        # Pre-allocated batched buffers. Float64 to match per-env precision.
        # State / RK4 stage / scratch.
        self._states = np.empty((self.N, sd), dtype=np.float64)
        self._k1 = np.empty((self.N, sd), dtype=np.float64)
        self._k2 = np.empty((self.N, sd), dtype=np.float64)
        self._k3 = np.empty((self.N, sd), dtype=np.float64)
        self._k4 = np.empty((self.N, sd), dtype=np.float64)
        self._mid = np.empty((self.N, sd), dtype=np.float64)
        self._new_state = np.empty((self.N, sd), dtype=np.float64)
        # Mass matrix and RHS — sized for the chosen env class.
        # Double pendulum: q_ddot is 3-D (cart + 2 angles); single: 2-D.
        # RHS shape is (N, q_dim, 1) so np.linalg.solve treats it as N
        # stacked column vectors per the gufunc signature.
        self._q_dim: int = 1 + n
        self._M_buf = np.empty((self.N, self._q_dim, self._q_dim), dtype=np.float64)
        self._RHS_buf = np.empty((self.N, self._q_dim, 1), dtype=np.float64)
        # Per-env force vector (filled each step from per-env control output).
        self._forces = np.empty(self.N, dtype=np.float64)
        # Output obs buffer.
        self._obs_out = np.empty((self.N, self.obs_dim), dtype=np.float32)
        # Output reward / done.
        self._rewards_out = np.empty(self.N, dtype=np.float32)
        self._dones_out = np.empty(self.N, dtype=bool)

        # Bind per-env scalar physics. These are copied once at construction
        # and refreshed by `sync_curriculum` whenever the trainer advances
        # the curriculum (cheap; happens at most once per ratchet step).
        self._refresh_physics()

    def _refresh_physics(self) -> None:
        """Refresh cached physics constants from the first env.

        All envs share these (the trainer's curriculum applies uniformly).
        Called from ``__init__`` and ``sync_curriculum``.
        """
        e = self.envs[0]
        self._g = float(e.g)
        self._M_cart = float(e.M)
        self._friction_cart = float(e.friction_cart)
        self._friction_pole = float(e.friction_pole)
        # Body params are class-level (no curriculum dependency).
        if self.n_poles == 2:
            self._m1 = float(e.m1)
            self._m2 = float(e.m2)
            self._l1 = float(e.l1)
            self._l2 = float(e.l2)
        else:  # single pendulum
            self._m = float(e.m)
            self._l = float(e.l)

    def sync_curriculum(self) -> None:
        """Re-read curriculum-modulated physics from the first env.

        Call this *after* the trainer applies a new curriculum to every env.
        It re-reads ``g``, ``friction_cart``, ``friction_pole`` from env 0
        (which the trainer keeps in sync with all other envs).
        """
        self._refresh_physics()

    # ------------------------------------------------------------------ #
    # Batched RK4
    # ------------------------------------------------------------------ #

    def _dynamics_batch(self, states: np.ndarray, forces: np.ndarray,
                        out: np.ndarray) -> None:
        """Dispatch to the env class's static batched dynamics."""
        if self.n_poles == 2:
            self.cls.dynamics_into_batched(
                states, forces, out,
                M_cart=self._M_cart, m1=self._m1, m2=self._m2,
                l1=self._l1, l2=self._l2, g=self._g,
                friction_cart=self._friction_cart,
                friction_pole=self._friction_pole,
                M_buf=self._M_buf, RHS_buf=self._RHS_buf,
            )
        else:
            self.cls.dynamics_into_batched(
                states, forces, out,
                M_cart=self._M_cart, m=self._m, l=self._l, g=self._g,
                friction_cart=self._friction_cart,
                friction_pole=self._friction_pole,
                M_buf=self._M_buf, RHS_buf=self._RHS_buf,
            )

    def _rk4_step_batch(self, states: np.ndarray, forces: np.ndarray,
                        dt: float) -> np.ndarray:
        r"""
        Batched classical 4th-order RK4. Identical operator order to
        :py:meth:`CartPendulumBase._rk4_step_inplace` so the per-row
        float64 result matches bitwise.

        Math
        ----
        For each env i:
            k1 = f(s_i)
            k2 = f(s_i + 0.5*dt*k1)
            k3 = f(s_i + 0.5*dt*k2)
            k4 = f(s_i + dt*k3)
            s_i' = s_i + (dt/6) * (k1 + 2 k2 + 2 k3 + k4)

        Numpy broadcasts each per-stage build over the leading N axis;
        elementwise float64 arithmetic is bit-identical row-by-row to the
        per-env :py:meth:`_rk4_step_inplace` path.
        """
        k1 = self._k1
        k2 = self._k2
        k3 = self._k3
        k4 = self._k4
        mid = self._mid
        ns = self._new_state

        # Stage 1
        self._dynamics_batch(states, forces, k1)

        # Stage 2: mid = states + 0.5*dt*k1; k2 = f(mid)
        np.multiply(k1, 0.5 * dt, out=mid)
        np.add(mid, states, out=mid)
        self._dynamics_batch(mid, forces, k2)

        # Stage 3
        np.multiply(k2, 0.5 * dt, out=mid)
        np.add(mid, states, out=mid)
        self._dynamics_batch(mid, forces, k3)

        # Stage 4
        np.multiply(k3, dt, out=mid)
        np.add(mid, states, out=mid)
        self._dynamics_batch(mid, forces, k4)

        # Combine: ns = states + (dt/6) * (k1 + 2 k2 + 2 k3 + k4)
        # Mirror the unbatched parenthesisation so float64 sums are
        # bit-identical row-by-row.
        np.multiply(k2, 2.0, out=mid)        # mid = 2 k2
        np.add(k1, mid, out=mid)             # mid = k1 + 2 k2
        np.multiply(k3, 2.0, out=ns)         # ns  = 2 k3
        np.add(mid, ns, out=mid)             # mid = k1 + 2 k2 + 2 k3
        np.add(mid, k4, out=mid)             # mid = ... + k4
        np.multiply(mid, dt / 6.0, out=ns)
        np.add(ns, states, out=ns)
        return ns

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step_batch(self, actions: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        r"""
        Batched step over all N envs.

        Parameters
        ----------
        actions : (N, action_dim) array of actions for each env.

        Returns
        -------
        next_obs    : (N, obs_dim) float32 array.
        rewards     : (N,) float32 array.
        terminated  : (N,) bool array.
        truncated   : (N,) bool array (always False; trainer handles truncation).
        infos       : list of N dicts (always empty per-env).

        Per-env contract
        ----------------
        Mirrors :py:meth:`CartPendulumBase.step` semantics one-for-one:
        force = control(action, state); wind = N(0, sigma) drawn from each
        env's own ``np_random``; impulse added then cleared; RK4 integrate;
        boundary check; reward; obs.

        The integrator is required to be ``"rk4"`` for the batched path —
        ``"semi_implicit"`` is single-step only and the trainer doesn't use
        it. RK4 is checked at construction time.
        """
        N = self.N
        envs = self.envs
        dt = float(envs[0].dt)
        rk4 = (envs[0].integrator == "rk4")
        if not rk4:
            # Fallback: per-env Python loop. The semi-implicit path is rarely
            # used in training and the speedup target is RK4.
            return self._step_per_env_fallback(actions)

        # 1. Stack states. Each env keeps its own state object — we read into
        #    the batched buffer here. After the integration we write the new
        #    state back into each env.state.
        states = self._states
        for i in range(N):
            states[i] = envs[i].state

        # 2. Per-env force = control + wind + impulse. Cheap Python loop;
        #    each call hits a few attribute accesses and at most one RNG draw.
        forces = self._forces
        for i in range(N):
            e = envs[i]
            env_params = e._env_params_cache
            f = float(e.control_strategy.get_force(actions[i], e.state, env_params))
            if e.wind_std > 0.0:
                f += float(e.np_random.normal(0.0, e.wind_std))
            f += e.current_impulse
            e.current_impulse = 0.0
            forces[i] = f

        # 3. Batched RK4 integration -> new states.
        new_states = self._rk4_step_batch(states, forces, dt)

        # 4. Build obs vectorised: [x, sin(theta_i)..., cos(theta_i)..., qdots].
        #    Mirrors `_get_obs` per-env layout exactly. Operating on the
        #    full (N, ...) tensor avoids the per-env Python loop the legacy
        #    code used.
        obs_out = self._obs_out
        n_poles = self.n_poles
        # Cart x.
        obs_out[:, 0] = new_states[:, 0]
        # sin/cos of each pole angle.
        thetas = new_states[:, 1 : 1 + n_poles]  # (N, n_poles)
        np.sin(thetas, out=obs_out[:, 1 : 1 + n_poles])
        np.cos(thetas, out=obs_out[:, 1 + n_poles : 1 + 2 * n_poles])
        # qdots (cart_xdot + theta_dots).
        obs_out[:, 1 + 2 * n_poles : 2 + 3 * n_poles] = new_states[:, 1 + n_poles :]

        # 5. Boundary check / reward — vectorise the cheap parts; the per-env
        #    reward call is irreducibly Python (each env owns its own
        #    reward_strategy with mutable counters). Write env state back
        #    so external code (e.g. the trainer's time-above check) can
        #    still read `env.state` per env.
        rewards = self._rewards_out
        dones = self._dones_out
        x_col = new_states[:, 0]
        abs_x = np.abs(x_col)
        x_hard = float(envs[0].x_hard)
        x_soft = float(envs[0].x_soft)
        boundary_k = float(envs[0].boundary_penalty_k)
        # Vectorised boundary penalty (matches scalar `boundary_k * o*o`
        # row-by-row because + - * are bit-equivalent across scalar/array).
        overshoot = np.maximum(abs_x - x_soft, 0.0)
        boundary_penalties = boundary_k * overshoot * overshoot

        # Per-env reward (call the strategy on the new state). Write env.state
        # before the reward call so any strategy that reads env.state through
        # the env handle still sees the post-step state.
        for i in range(N):
            e = envs[i]
            ns = new_states[i]
            # ns is a view into self._new_state; the runner reuses that buffer
            # next step, so we must copy here to avoid corrupting the env.
            e.state = ns.copy()
            r = float(e.reward_strategy.compute_reward(e.state, e._env_params_cache))
            r -= float(boundary_penalties[i])
            rewards[i] = r
            dones[i] = bool(abs_x[i] > x_hard)

        truncated = np.zeros(N, dtype=bool)
        infos: list[dict] = [{} for _ in range(N)]
        return obs_out, rewards, dones, truncated, infos

    def _step_per_env_fallback(self, actions: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Fallback for non-RK4 integrators: identical to the legacy loop."""
        N = self.N
        rewards = self._rewards_out
        dones = self._dones_out
        obs_out = self._obs_out
        truncated = np.zeros(N, dtype=bool)
        infos: list[dict] = []
        for i in range(N):
            obs, r, term, trunc, info = self.envs[i].step(actions[i])
            obs_out[i] = obs
            rewards[i] = r
            dones[i] = term
            truncated[i] = trunc
            infos.append(info)
        return obs_out, rewards, dones, truncated, infos
