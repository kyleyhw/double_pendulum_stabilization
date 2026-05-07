r"""
Soft Actor-Critic (SAC) for continuous control.

Why SAC here
============
Phase J of the campaign log proved that state-dependent :math:`\log\sigma`
is incompatible with PPO: under the importance-ratio gradient, when
:math:`\sigma_\theta(s)` shifts between rollout collection and update the
ratio variance explodes and KL grows unbounded. SAC's *reparameterised*
gradient propagates through both the mean and the log-std heads cleanly,
making the same architecture stable. Under PPO the 4-7% strict-success
ceiling at :math:`\delta \approx 0.45` is the structural floor; SAC's
state-dependent variance is the textbook escape.

Mathematical structure
======================
Let :math:`\pi_\theta(a \mid s)` be a tanh-squashed Gaussian with state-
dependent mean :math:`\mu_\theta(s)` and log-std
:math:`\log\sigma_\theta(s)`, and :math:`Q_{\phi_i}(s, a)` be twin critic
networks (i = 1, 2). SAC optimises the maximum-entropy objective

.. math::

    J(\pi) = \sum_t \mathbb E_{(s_t, a_t) \sim \rho_\pi}
        \bigl[ r(s_t, a_t) + \alpha \mathcal H(\pi(\cdot \mid s_t)) \bigr],

with three update rules:

* **Critic** (off-policy TD with entropy bonus and twin-Q minimum):

    .. math::

        y = r + \gamma (1 - d) \bigl[ \min_i Q_{\bar\phi_i}(s', \tilde a')
            - \alpha \log\pi_\theta(\tilde a' \mid s') \bigr],
        \qquad \tilde a' \sim \pi_\theta(\cdot \mid s'),

  with :math:`\bar\phi_i` the Polyak-averaged target params. Loss:
  :math:`\sum_i (Q_{\phi_i}(s, a) - y)^2`. The min over twin Q's
  controls overestimation bias from naive bootstrapping (Hasselt et al.).

* **Actor** (reparameterised gradient through the squashed Gaussian):

    .. math::

        \mathcal L_\pi = \mathbb E_{s \sim \mathcal D}
            \Bigl[ \alpha \log\pi_\theta(\tilde a \mid s)
                 - \min_i Q_{\phi_i}(s, \tilde a) \Bigr],
        \qquad \tilde a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \xi),
        \quad \xi \sim \mathcal N(0, I).

  The reparameterisation lets gradients flow into both heads of the
  actor — the exact thing PPO cannot do.

* **Temperature** (automatic entropy tuning, Haarnoja et al. 2018b):

    .. math::

        \mathcal L_\alpha = \mathbb E_{s \sim \mathcal D}
            \bigl[ -\alpha \bigl( \log\pi_\theta(\tilde a \mid s)
                                + \mathcal H_{\rm target} \bigr) \bigr],

  with target entropy :math:`\mathcal H_{\rm target} = -|\mathcal A|`
  (one nat per action dim by convention). Optimised over :math:`\log\alpha`
  with Adam to keep :math:`\alpha > 0` automatically.

Off-policy: a uniform replay buffer of capacity 1M stores transitions
:math:`(s, a, r, s', d)`. Each gradient step samples a minibatch of
256 from the buffer.

Polyak averaging: target nets updated as
:math:`\bar\phi_i \leftarrow \tau \phi_i + (1 - \tau) \bar\phi_i` with
:math:`\tau = 0.005` (slow tracking — stabilises the TD target).

Tanh log-prob correction
========================
For :math:`a = \tanh(z)` with :math:`z \sim \mathcal N(\mu, \sigma)`,
the change-of-variables yields

.. math::

    \log\pi(a \mid s) = \log\mathcal N(z \mid \mu, \sigma)
        - \sum_i \log\bigl(1 - \tanh^2 z_i + \varepsilon\bigr).

The numerically stable form ``2*(log(2) - z - softplus(-2z))`` is used
for the squash term to avoid catastrophic cancellation when :math:`|z|`
is large.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0
SQUASH_EPS: float = 1e-6


# ---------------------------------------------------------------------------- #
# Networks
# ---------------------------------------------------------------------------- #

class GaussianPolicy(nn.Module):
    r"""
    Tanh-squashed Gaussian policy with state-dependent log-std.

    The trunk is a 2-layer Tanh MLP (matching the existing PPO actor for
    architectural parity); the head outputs ``2 * action_dim`` values
    split into mean and log-std.

    The reparameterised sample :math:`a = \tanh(z),
    z = \mu + \sigma \cdot \xi` is differentiable in both :math:`\mu` and
    :math:`\sigma` — this is what SAC needs and PPO cannot exploit.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, *, deterministic: bool = False
               ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Reparameterised sample plus log-prob with the tanh Jacobian
        correction.

        Returns ``(action, log_prob)``. ``deterministic=True`` skips the
        noise term and returns :math:`\tanh(\mu)` (used at inference).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        if deterministic:
            z = mean
        else:
            xi = torch.randn_like(mean)
            z = mean + std * xi
        action = torch.tanh(z)

        # Pre-squash log-prob
        log_prob_z = -0.5 * (
            ((z - mean) / std).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        # Tanh Jacobian correction. Use the numerically stable identity
        # log(1 - tanh^2 z) = 2 * (log 2 - z - softplus(-2z)).
        squash_correction = 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
        log_prob = (log_prob_z - squash_correction).sum(dim=-1)
        return action, log_prob


class TwinQ(nn.Module):
    r"""
    Two independent Q-networks :math:`Q_1(s, a)` and :math:`Q_2(s, a)`.

    Per the SAC paper, ``forward`` returns both Q-values. The TD target
    uses ``min(Q1', Q2')`` to control overestimation bias.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = self._build_q(state_dim, action_dim, hidden_dim)
        self.q2 = self._build_q(state_dim, action_dim, hidden_dim)

    @staticmethod
    def _build_q(state_dim: int, action_dim: int, hidden_dim: int) -> nn.Sequential:
        net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(net[-1].weight, gain=1.0)
        nn.init.zeros_(net[-1].bias)
        return net

    def forward(self, state: torch.Tensor, action: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ---------------------------------------------------------------------------- #
# Replay buffer (CPU-resident, on-device transfer at sample time)
# ---------------------------------------------------------------------------- #

@dataclass
class ReplayBuffer:
    r"""
    Uniform-sampling FIFO replay buffer for SAC.

    Storage is CPU numpy float32 (cheap memory; 1M transitions on a
    7-D obs + 1-D action buffer ~ 50 MB). Sampled minibatches are
    moved to the agent's device once per update.
    """
    capacity: int
    state_dim: int
    action_dim: int

    states: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    next_states: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    size: int = field(default=0, init=False)
    pos: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        c = int(self.capacity)
        self.states = np.zeros((c, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((c, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(c, dtype=np.float32)
        self.next_states = np.zeros((c, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(c, dtype=np.float32)

    def push_batch(self, states: np.ndarray, actions: np.ndarray,
                   rewards: np.ndarray, next_states: np.ndarray,
                   dones: np.ndarray) -> None:
        """Append a batch of transitions; wraps around at capacity."""
        n = states.shape[0]
        idx = (self.pos + np.arange(n)) % self.capacity
        self.states[idx] = states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_states
        self.dones[idx] = dones
        self.pos = int((self.pos + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def sample(self, batch_size: int, *, device: torch.device
               ) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


# ---------------------------------------------------------------------------- #
# Agent
# ---------------------------------------------------------------------------- #

class SACAgent:
    r"""
    Soft Actor-Critic with twin-Q, target nets, and automatic entropy tuning.

    Defaults follow Haarnoja et al. (2018b):

    * :math:`\gamma = 0.99`
    * :math:`\tau = 0.005` (Polyak coefficient)
    * Replay buffer capacity 1M, batch size 256
    * Target entropy :math:`\mathcal H_{\rm target} = -|\mathcal A|`
    * Adam at :math:`3 \times 10^{-4}` for actor, twin-Q, and
      :math:`\log\alpha`
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        batch_size: int = 256,
        replay_capacity: int = 1_000_000,
        target_entropy: Optional[float] = None,
        init_log_alpha: float = 0.0,
        alpha_max: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.target_entropy = float(target_entropy) if target_entropy is not None \
            else -float(action_dim)
        # Optional upper clamp on alpha. When the curriculum advances and the
        # replay buffer briefly contains stale low-difficulty transitions,
        # auto-entropy can ramp alpha unbounded (Phase N continuation 3
        # observed alpha climbing to 1.0+ at d=0.295), which adds enough
        # noise to the stochastic policy that the curriculum gate's
        # `time_above` measurement falls below the threshold even though
        # the deterministic policy is competent. Clamping alpha at ~0.3 keeps
        # the entropy bonus meaningful for exploration without drowning the
        # actor's precision near upright.
        self._alpha_max: Optional[float] = (
            float(alpha_max) if alpha_max is not None else None
        )
        self._log_alpha_max: Optional[float] = (
            float(math.log(self._alpha_max)) if self._alpha_max is not None else None
        )

        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = TwinQ(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinQ(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.log_alpha = torch.tensor(float(init_log_alpha),
                                      device=self.device, requires_grad=True)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        self.buffer = ReplayBuffer(replay_capacity, state_dim, action_dim)

        self._last_diagnostics: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # --- inference ---------------------------------------------------------- #

    def select_action(self, state: np.ndarray, *, deterministic: bool = False
                      ) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            squeeze = (s.ndim == 1)
            if squeeze:
                s = s.unsqueeze(0)
            action, _ = self.actor.sample(s, deterministic=deterministic)
            if squeeze:
                action = action.squeeze(0)
            return action.cpu().numpy()

    # --- updates ------------------------------------------------------------ #

    def update(self) -> dict[str, float]:
        r"""
        Single SAC update step. Samples a minibatch from the replay buffer
        and runs critic / actor / temperature gradient steps in that order
        (matching the reference impl).
        """
        if self.buffer.size < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, device=self.device,
        )

        # ----- Critic update --------------------------------------------------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target = rewards + self.gamma * (1.0 - dones) * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ----- Actor update ---------------------------------------------------
        new_action, log_prob = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----- Temperature update --------------------------------------------
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # Clamp log_alpha from above if requested (prevents runaway exploration
        # when the gradient pushes alpha unbounded — see __init__ doc).
        if self._log_alpha_max is not None:
            with torch.no_grad():
                self.log_alpha.clamp_(max=self._log_alpha_max)

        # ----- Polyak averaging on target critic ------------------------------
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(),
                             self.critic.parameters()):
                tp.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        diag = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "log_prob_mean": float(log_prob.mean().item()),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "target_mean": float(target.mean().item()),
        }
        self._last_diagnostics = diag
        return diag

    # --- checkpoint --------------------------------------------------------- #

    def save(self, path: str, *, extra: Optional[dict[str, Any]] = None) -> None:
        payload: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        return payload
