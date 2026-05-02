r"""
Proximal Policy Optimization with:

* GAE-:math:`\lambda` advantage estimation.
* Tanh-squashed Gaussian policy with the standard log-Jacobian correction
  to the log-probability — eliminates the bias that arises from clipping a
  Normal distribution into :math:`[-1, 1]` (the original implementation's
  bias became severe at :math:`\sigma \gtrsim 0.3`).
* Minibatched K-epoch updates with shuffling.
* Learning-rate, entropy-coefficient, and clip-range schedules driven by
  training progress :math:`p \in [0, 1]`.
* Gradient norm clipping.
* Checkpoint state-dict packs both the policy weights and the running
  observation normaliser, so inference and training stay tied.

Mathematical notes
==================
With :math:`z \sim \mathcal N(\mu_\theta(s), \sigma_\theta)` and
:math:`a = \tanh z`, the change-of-variables formula yields

.. math::

    \log \pi(a \mid s) = \log \mathcal N(z \mid \mu, \sigma) - \sum_i
    \log\!\bigl(1 - \tanh^2 z_i + \varepsilon\bigr).

We sample :math:`z` and apply the squash; reparameterisation is unnecessary
because PPO uses the importance ratio of stored log-probabilities (it does
not backpropagate through the sampling).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ---------------------------------------------------------------------------- #
# Networks
# ---------------------------------------------------------------------------- #

class ActorCritic(nn.Module):
    r"""
    Two-headed MLP. Returns a tanh-squashed Gaussian policy and a state-value.

    The squashed distribution provides actions natively in :math:`[-1, 1]`,
    matching the action box used by both control strategies, with no clipping
    bias in the log-prob.

    Two policy variants are supported (controlled by ``state_dependent_std``):

    * ``False`` (default): a single learnable scalar :math:`\log\sigma` per
      action dim, shared across all states. Simpler, fewer parameters, but
      cannot express "small noise near upright, big noise during swing-up".

    * ``True``: the actor produces ``2 * action_dim`` outputs split into a
      mean head and a log-std head, both state-dependent. The log-std head
      shares the trunk with the mean head so the policy can shrink its noise
      where the value function says it should (e.g. near the LQR-style
      optimum). Output is clamped to ``[LOG_STD_MIN, LOG_STD_MAX]``.

    The state-dependent variant is necessary when the optimal policy
    requires *micro*-corrections (LQR-style stabilisation) that cannot
    survive a state-independent noise floor of :math:`\sigma \approx 0.5`.
    Empirically, this is the bottleneck once the reward shape gives a
    gradient toward tight stabilisation (Phase H3 onwards).
    """
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 2.0
    SQUASH_EPS: float = 1e-6

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 init_log_std: float = -0.5, state_dependent_std: bool = False) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.state_dependent_std = bool(state_dependent_std)

        actor_out = action_dim * (2 if self.state_dependent_std else 1)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, actor_out),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)

        self.actor.apply(_init)
        self.critic.apply(_init)

        # Final actor layer: small weights -> near-zero initial mean.
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

        if self.state_dependent_std:
            # Bias the log-std head's bias term so the initial output is
            # init_log_std (roughly) for any input, since the orthogonal
            # weight init has gain 0.01 on the final layer.
            with torch.no_grad():
                self.actor[-1].bias[self.action_dim:].fill_(float(init_log_std))
            self.log_std = None  # not used in state-dependent mode
        else:
            self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_log_std)))

    def _split_actor_output(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) from the actor output (state-dependent path)."""
        mean = raw[..., : self.action_dim]
        log_std = raw[..., self.action_dim :].clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def _dist(self, state: torch.Tensor, min_std: float = 0.0) -> Normal:
        raw = self.actor(state)
        if self.state_dependent_std:
            mean, log_std = self._split_actor_output(raw)
            std = log_std.exp()
        else:
            mean = raw
            log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = log_std.exp().expand_as(mean)
        if min_std > 0.0:
            std = torch.clamp(std, min=min_std)
        return Normal(mean, std)

    def get_action(self, state: torch.Tensor, *, deterministic: bool = False,
                   min_std: float = 0.0) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        dist = self._dist(state, min_std=min_std)
        if deterministic:
            z = dist.mean
            a = torch.tanh(z)
            return a, None
        z = dist.rsample()
        a = torch.tanh(z)
        # log p(a) = log p(z) - sum log(1 - tanh(z)^2)
        log_prob = dist.log_prob(z).sum(dim=-1) - torch.log(1.0 - a.pow(2) + self.SQUASH_EPS).sum(dim=-1)
        return a, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Recover pre-squash z from stored a via atanh, then evaluate.
        a_clamped = action.clamp(-1.0 + self.SQUASH_EPS, 1.0 - self.SQUASH_EPS)
        z = torch.atanh(a_clamped)
        # _dist handles the state-dependent vs state-independent split.
        dist = self._dist(state)
        log_prob = dist.log_prob(z).sum(dim=-1) - torch.log(1.0 - a_clamped.pow(2) + self.SQUASH_EPS).sum(dim=-1)
        # Differential entropy of tanh-squashed Gaussian has no closed form;
        # return the pre-squash entropy as a usable surrogate (monotone in sigma).
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        return log_prob, value, entropy


# ---------------------------------------------------------------------------- #
# Memory
# ---------------------------------------------------------------------------- #

@dataclass
class Memory:
    """Rollout buffer for one PPO update window."""
    states: list = None
    actions: list = None
    log_probs: list = None
    rewards: list = None
    is_terminals: list = None

    def __post_init__(self) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


# ---------------------------------------------------------------------------- #
# PPO agent
# ---------------------------------------------------------------------------- #

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        lr: float | Callable[[float], float] = 3e-4,
        gamma: float = 0.99,
        eps_clip: float | Callable[[float], float] = 0.2,
        k_epochs: int = 10,
        gae_lambda: float = 0.95,
        entropy_coef: float | Callable[[float], float] = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        init_log_std: float = -0.5,
        hidden_dim: int = 256,
        minibatch_size: int = 256,
        target_kl: Optional[float] = 0.03,
        value_clip: Optional[float] = None,
        state_dependent_std: bool = False,
    ) -> None:
        self.gamma = float(gamma)
        self.k_epochs = int(k_epochs)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.minibatch_size = int(minibatch_size)
        self.target_kl = target_kl
        # If set, value loss uses PPO2-style clipping with this half-width:
        #   V_clip = V_old + clip(V_new - V_old, +/- value_clip);
        #   loss   = mean( max( (V_new - R)^2, (V_clip - R)^2 ) )
        # Prevents the value head from following the policy too aggressively
        # when the policy is making a destructive update.
        self.value_clip = None if value_clip is None else float(value_clip)

        # Coerce scalars to schedule callables.
        self._lr_fn = lr if callable(lr) else (lambda _p, v=lr: float(v))
        self._eps_fn = eps_clip if callable(eps_clip) else (lambda _p, v=eps_clip: float(v))
        self._ent_fn = entropy_coef if callable(entropy_coef) else (lambda _p, v=entropy_coef: float(v))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim=hidden_dim,
                                  init_log_std=init_log_std,
                                  state_dependent_std=state_dependent_std).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self._lr_fn(0.0))
        self.loss_fn = nn.MSELoss()

        self._last_diagnostics: dict[str, float] = {}

    # --- inference helpers ---------------------------------------------------

    def select_action(self, state: np.ndarray, *, deterministic: bool = False,
                      min_std: float = 0.0) -> tuple[np.ndarray, Optional[np.ndarray]]:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if s.ndim == 1:
                s = s.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            action, log_prob = self.policy.get_action(s, deterministic=deterministic, min_std=min_std)
            if squeeze:
                action = action.squeeze(0)
                if log_prob is not None:
                    log_prob = log_prob.squeeze(0)
        np_action = action.cpu().numpy()
        np_lp = None if log_prob is None else log_prob.cpu().numpy()
        return np_action, np_lp

    def get_value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return float(self.policy.critic(s).squeeze().item())

    # --- GAE helper ----------------------------------------------------------

    def compute_gae(
        self,
        rewards: np.ndarray,
        is_terminals: np.ndarray,
        states: np.ndarray,
        last_state: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Generalized Advantage Estimation for a single trajectory of length :math:`n`.

        Returns ``(advantages, returns)`` of shape ``(n,)``. Bootstraps the tail
        with :math:`V_\phi(s_{n})` (passed as ``last_state``) unless the trajectory
        ended in a terminal step.
        """
        n = len(rewards)
        if n == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        with torch.no_grad():
            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            old_values = self.policy.critic(states_t).squeeze(-1).cpu().numpy().astype(np.float32)
            if last_state is not None and not is_terminals[-1]:
                last_t = torch.as_tensor(last_state, dtype=torch.float32, device=self.device)
                last_v = float(self.policy.critic(last_t).squeeze().item())
            else:
                last_v = 0.0

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = last_v
        for t in range(n - 1, -1, -1):
            mask = 0.0 if is_terminals[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * mask - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = old_values[t]

        returns = advantages + old_values
        return advantages, returns

    def update_from_arrays(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        progress: float = 0.0,
    ) -> dict[str, float]:
        """PPO update from pre-computed advantages / returns (vectorised path)."""
        n = states.shape[0]
        if n == 0:
            return {}
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_lp_t = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return self._k_epoch_update(states_t, actions_t, old_lp_t, adv_t, ret_t,
                                    n=n, progress=progress, returns_np=returns)

    # --- update (single-trajectory path) -------------------------------------

    def update(self, memory: Memory, *, last_state: Optional[np.ndarray] = None,
               progress: float = 0.0) -> dict[str, float]:
        n = len(memory)
        if n == 0:
            return {}
        rewards = np.asarray(memory.rewards, dtype=np.float32)
        is_terminals = np.asarray(memory.is_terminals, dtype=bool)
        states = np.asarray(memory.states, dtype=np.float32)
        actions = np.asarray(memory.actions, dtype=np.float32)
        log_probs = np.asarray(memory.log_probs, dtype=np.float32)
        advantages, returns = self.compute_gae(rewards, is_terminals, states, last_state=last_state)
        return self.update_from_arrays(states, actions, log_probs, advantages, returns, progress=progress)

    def _k_epoch_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        adv_t: torch.Tensor,
        ret_t: torch.Tensor,
        *,
        n: int,
        progress: float,
        returns_np: np.ndarray,
    ) -> dict[str, float]:
        old_log_probs_t = old_log_probs

        # Apply schedules.
        lr = self._lr_fn(progress)
        eps_clip = self._eps_fn(progress)
        ent_coef = self._ent_fn(progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Pre-update value baseline for the explained-variance metric and
        # (when value_clip is set) the per-minibatch reference for clipped MSE.
        with torch.no_grad():
            old_values_full = self.policy.critic(states).squeeze(-1).detach()
            old_values_np = old_values_full.cpu().numpy().astype(np.float32)

        idxs = np.arange(n)
        approx_kls: list[float] = []
        clip_fracs: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        early_stop = False

        for _ in range(self.k_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.minibatch_size):
                mb = idxs[start : start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                log_probs, values, entropy = self.policy.evaluate(states[mb_t], actions[mb_t])
                ratios = torch.exp(log_probs - old_log_probs_t[mb_t])
                surr1 = ratios * adv_t[mb_t]
                surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * adv_t[mb_t]

                policy_loss = -torch.min(surr1, surr2).mean()

                if self.value_clip is not None:
                    v_old = old_values_full[mb_t]
                    v_clipped = v_old + torch.clamp(values - v_old, -self.value_clip, self.value_clip)
                    vl_unclipped = (values - ret_t[mb_t]).pow(2)
                    vl_clipped = (v_clipped - ret_t[mb_t]).pow(2)
                    value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
                else:
                    value_loss = self.loss_fn(values, ret_t[mb_t])

                entropy_bonus = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - ent_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kls.append(float(((ratios - 1.0) - (log_probs - old_log_probs_t[mb_t])).mean().item()))
                    clip_fracs.append(float((torch.abs(ratios - 1.0) > eps_clip).float().mean().item()))
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy_bonus.item()))

            if self.target_kl is not None and approx_kls and approx_kls[-1] > 1.5 * self.target_kl:
                early_stop = True
                break

        explained_variance = float(
            1.0 - np.var(returns_np - old_values_np) / (np.var(returns_np) + 1e-8)
        )

        diag = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": explained_variance,
            "lr": lr,
            "eps_clip": eps_clip,
            "entropy_coef": ent_coef,
            "early_stopped": float(early_stop),
        }
        self._last_diagnostics = diag
        return diag

    # --- checkpoint I/O ------------------------------------------------------

    def save(self, checkpoint_path: str, *, extra: Optional[dict[str, Any]] = None) -> None:
        payload: dict[str, Any] = {"policy": self.policy.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, checkpoint_path)

    def load(self, checkpoint_path: str) -> dict[str, Any]:
        # weights_only=False is required because our checkpoints package numpy
        # arrays (the running observation normaliser state) alongside the policy
        # weights. We only ever load checkpoints written by this codebase.
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(payload, dict) and "policy" in payload:
            self.policy.load_state_dict(payload["policy"])
            return payload
        # Backwards compatibility: bare state_dict.
        self.policy.load_state_dict(payload)
        return {"policy": payload}
