r"""
Observation normalisation via running mean / variance (Welford updates).

The wrapper :class:`NormalizeObservation` standardises each observation
component to zero mean and unit variance using a running estimate maintained
in :class:`RunningMeanStd`. The estimate is a fixed function of all
observations seen so far; freezing it after a calibration phase guarantees
stationary inputs to the policy.

Welford update (vector form). For new sample :math:`x \in \mathbb R^d`,

.. math::

    \begin{aligned}
    \delta &= x - \mu_n,\\
    \mu_{n+1} &= \mu_n + \delta / (n+1),\\
    M^{(2)}_{n+1} &= M^{(2)}_n + \delta \odot (x - \mu_{n+1}),\\
    \sigma^2_{n+1} &= M^{(2)}_{n+1} / n_{n+1}.
    \end{aligned}

Numerically stable for arbitrary :math:`n`; no catastrophic cancellation.

The state of an :class:`RunningMeanStd` is fully serialisable via
:py:meth:`state_dict` so it can be saved alongside the policy checkpoint and
restored at inference time. Mismatched normalisation between training and
inference is a common silent source of degraded policies — keep these tied.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class RunningMeanStd:
    """Online mean / variance estimator (per-component, vector-valued)."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == self.mean.ndim:
            x = x[None, ...]  # treat as a 1-sample batch
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = float(x.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray,
                             batch_count: float) -> None:
        # Parallel-Welford merge (Chan et al. 1979).
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class NormalizeObservation(gym.Wrapper):
    r"""
    Standardise observations using a running mean/var estimate.

    During training, ``training`` is True and the wrapper updates the estimate
    after each observation. At inference time, set ``training = False`` to
    freeze the normaliser. The :class:`RunningMeanStd` instance is exposed as
    ``self.obs_rms`` for serialisation.

    The clipping bound :math:`c` (default 10) caps :math:`\hat o = \mathrm{clip}((o - \mu)/\sigma, \pm c)`
    to bound the effect of any single outlier on the policy's input distribution.
    """

    def __init__(self, env: gym.Env, *, epsilon: float = 1e-8, clip: float = 10.0,
                 training: bool = True) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), \
            "NormalizeObservation only supports Box observation spaces."
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.training: bool = bool(training)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.training:
            self.obs_rms.update(obs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.training:
            self.obs_rms.update(obs)
        return self._normalize(obs), reward, terminated, truncated, info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        out = (np.asarray(obs, dtype=np.float64) - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )
        return np.clip(out, -self.clip, self.clip).astype(np.float32)


# ---------------------------------------------------------------------------- #
# Batched application helper (candidate-3 trainer fast-path)
# ---------------------------------------------------------------------------- #

def batched_normalize(obs_batch: np.ndarray, rms: RunningMeanStd, *,
                      epsilon: float = 1e-8, clip: float = 10.0) -> np.ndarray:
    r"""Vectorised application of :class:`NormalizeObservation` over an N-env batch.

    Parameters
    ----------
    obs_batch
        ``(N, obs_dim)`` array of raw observations from N parallel envs.
    rms
        Shared :class:`RunningMeanStd` instance providing mean and variance.

    Returns
    -------
    ``(N, obs_dim)`` array, dtype float32, clipped to ``[-clip, clip]``.

    Math
    ----
    The per-env path computed
    :math:`\hat o_i = \mathrm{clip}((o_i - \mu)/\sqrt{\sigma^2 + \varepsilon}, \pm c)`
    one row at a time. Since the formula is element-wise, batching gives an
    identical numerical result (subject to the same float64-then-float32 cast).
    The single broadcast op also has lower Python overhead than N wrapper calls.
    """
    diff = np.asarray(obs_batch, dtype=np.float64) - rms.mean
    np.divide(diff, np.sqrt(rms.var + epsilon), out=diff)
    np.clip(diff, -clip, clip, out=diff)
    return diff.astype(np.float32)


def batched_rms_update(rms: RunningMeanStd, obs_batch: np.ndarray) -> None:
    r"""Single Welford-merge update for an N-env batch.

    Equivalent to applying :py:meth:`RunningMeanStd.update` once per row, but
    folds the N rows into a single merge step. The parallel-Welford merge is
    associative *up to floating-point rounding* — for typical N ~ 8 the
    difference is at the float64 ULP level, well within the noise the
    normaliser already absorbs (it converges in distribution to the true
    statistics regardless of merge order).
    """
    rms.update(np.asarray(obs_batch, dtype=np.float64))
