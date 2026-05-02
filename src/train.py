r"""
PPO trainer with curriculum, vectorised rollouts, observation normalisation,
and TensorBoard logging.

High-level pipeline
===================

1. Build :math:`N` parallel envs via :class:`gymnasium.vector.SyncVectorEnv`.
   The same curriculum :math:`\delta` is applied to every env.

2. Wrap with :class:`NormalizeObservation`. The running mean/var is updated
   from rollout data and frozen at evaluation/inference.

3. Collect :math:`T` steps from each env per PPO update — total batch
   :math:`T \cdot N`. Bootstraps the value of the post-rollout state to keep
   GAE correct on truncated rollouts.

4. PPO update with minibatches, schedule-driven LR / clip / entropy.

5. Curriculum advance (the **adaptive ratchet**): each level-up step
   :math:`\Delta\delta` is proportional to the recent improvement
   :math:`(\bar R - \bar R_{\rm prev best})/R_{\max}`, capped at
   :math:`\Delta\delta_{\max}`. This lets the agent leap on big wins and
   creep on small ones, instead of always advancing by 1%.

6. Periodic deterministic-policy evaluation episodes for an unbiased
   measurement of policy quality.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch

# Add project root to path.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.ppo import Memory, PPOAgent  # noqa: E402
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.env.single_pendulum import SinglePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl, VelocityControl  # noqa: E402
from src.strategies.rewards import (  # noqa: E402
    DoublePendulumStandardReward,
    EnergyShapingReward,
    ExponentialSwingUpReward,
    HybridLQRSwingUpReward,
    LQRCostReward,
    RewardStrategy,
    SinglePendulumStandardReward,
)
from src.utils.normalize import NormalizeObservation  # noqa: E402
from src.utils.schedules import linear  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tb is optional
    SummaryWriter = None


# ---------------------------------------------------------------------------- #
# Factories
# ---------------------------------------------------------------------------- #

def _build_reward(args: argparse.Namespace) -> RewardStrategy:
    curve = getattr(args, "threshold_curve", "linear")
    if args.reward == "exponential":
        return ExponentialSwingUpReward(threshold_curve=curve)
    if args.reward == "standard":
        return (SinglePendulumStandardReward(survival_alpha=args.survival_alpha,
                                             threshold_curve=curve)
                if args.env == "single" else
                DoublePendulumStandardReward(survival_alpha=args.survival_alpha,
                                             threshold_curve=curve))
    if args.reward == "energy":
        return EnergyShapingReward(threshold_curve=curve)
    if args.reward == "lqr":
        return LQRCostReward()
    if args.reward == "hybrid":
        return HybridLQRSwingUpReward(threshold_curve=curve)
    raise ValueError(f"Unknown reward strategy: {args.reward}")


def _build_env(args: argparse.Namespace, seed: int) -> Any:
    """Build a single un-wrapped env. Each call returns a fresh instance with its own state."""
    control = ForceControl() if args.control == "force" else VelocityControl()
    reward = _build_reward(args)
    cls = SinglePendulumCartEnv if args.env == "single" else DoublePendulumCartEnv
    env = cls(
        reset_mode=args.reset_mode,
        control_strategy=control,
        reward_strategy=reward,
        integrator=args.integrator,
        x_soft=args.x_soft,
        x_hard=args.x_hard,
        boundary_penalty_k=args.boundary_penalty_k,
        wind_max=args.wind_max,
    )
    env.reset(seed=seed)
    return env


def _make_vector_env(args: argparse.Namespace, n_envs: int, seed: int):
    import gymnasium as gym

    def _factory(i: int):
        def _f():
            return _build_env(args, seed=seed + i)
        return _f

    venv = gym.vector.SyncVectorEnv([_factory(i) for i in range(n_envs)])
    return venv


def _max_episode_reward(reward_strategy: RewardStrategy, max_steps: int, dt: float) -> float:
    if isinstance(reward_strategy, ExponentialSwingUpReward):
        steps = np.arange(1, max_steps + 1, dtype=np.float64)
        per = np.exp(np.minimum(steps * dt, reward_strategy.t_cap)) - 1.0
        return float(per.sum())
    if isinstance(reward_strategy, EnergyShapingReward):
        return float(reward_strategy.max_per_step_reward() * max_steps)
    if isinstance(reward_strategy, LQRCostReward):
        return float(reward_strategy.max_per_step_reward() * max_steps)
    if isinstance(reward_strategy, HybridLQRSwingUpReward):
        # Hybrid is bounded above by the exp reward's max (penalty terms only subtract).
        steps = np.arange(1, max_steps + 1, dtype=np.float64)
        per = np.exp(np.minimum(steps * dt, reward_strategy._exp.t_cap)) - 1.0
        return float(per.sum())
    return 1.5 * max_steps


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def _set_curriculum_all(venv, normaliser_envs: list, difficulty: float) -> None:
    """Apply curriculum to every underlying env in the vector."""
    for env in normaliser_envs:
        # `env` here is the NormalizeObservation wrapper; its .env is the raw env.
        env.env.set_curriculum(difficulty)


def _angles_from_obs(obs: np.ndarray, env_kind: str) -> tuple[float, ...]:
    """Reconstruct angles from a (possibly normalised) observation. Use raw obs."""
    if env_kind == "double":
        s1, c1 = obs[1], obs[3]
        s2, c2 = obs[2], obs[4]
        return float(np.arctan2(s1, c1)), float(np.arctan2(s2, c2))
    s1, c1 = obs[1], obs[2]
    return (float(np.arctan2(s1, c1)),)


def _evaluate(env_args: argparse.Namespace, agent: PPOAgent, obs_rms_state: dict,
              difficulty: float, n_episodes: int, max_steps: int, seed: int) -> dict[str, float]:
    """Run deterministic-policy episodes for unbiased evaluation."""
    raw_env = _build_env(env_args, seed=seed)
    raw_env.set_curriculum(difficulty)
    eval_env = NormalizeObservation(raw_env, training=False)
    eval_env.obs_rms.load_state_dict(obs_rms_state)

    rewards = []
    times_above = []
    lengths = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + 1000 + ep)
        ep_r = 0.0
        steps_above = 0
        t = 0
        for t in range(max_steps):
            action, _ = agent.select_action(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(action)
            ep_r += float(r)
            # Time-above metric uses raw angles from the underlying env state.
            theta_errs = [
                abs(np.arctan2(np.sin(a - np.pi), np.cos(a - np.pi)))
                for a in eval_env.env.state[1:1 + raw_env.n_poles]
            ]
            if all(e < raw_env.reward_strategy.reward_threshold for e in theta_errs):
                steps_above += 1
            if term or trunc:
                break
        rewards.append(ep_r)
        times_above.append(steps_above / max_steps)
        lengths.append(t + 1)

    return {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/time_above_mean": float(np.mean(times_above)),
        "eval/length_mean": float(np.mean(lengths)),
    }


# ---------------------------------------------------------------------------- #
# Main loop
# ---------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    seed = int(args.seed) if args.seed is not None else int(np.random.randint(0, 100_000))
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"[seed]   {seed}")
    print(f"[config] env={args.env} control={args.control} reward={args.reward} "
          f"integrator={args.integrator} n_envs={args.n_envs}")

    # --- envs ---
    venv = _make_vector_env(args, args.n_envs, seed)
    # Wrap each env with its own normaliser sharing the SAME RunningMeanStd.
    # SyncVectorEnv wraps multiple env factories; we need to access them post-build.
    # Simplest approach: wrap the vector env with a single NormalizeObservation
    # that operates on the batched obs. Implement inline:
    raw_envs = [venv.envs[i] for i in range(args.n_envs)]
    norm_envs = [NormalizeObservation(e, training=True) for e in raw_envs]

    # The first env's normaliser is the canonical one; share its state across all wrappers.
    canonical_rms = norm_envs[0].obs_rms
    for w in norm_envs[1:]:
        w.obs_rms = canonical_rms

    # Hyperparameters.
    max_timesteps = args.episode_steps      # per-episode cap (also rollout cap per env)
    rollout_steps = args.rollout_steps      # steps per env per PPO update
    total_updates = args.updates

    # Schedules.
    lr_schedule = linear(args.lr, args.lr * args.lr_final_frac)
    ent_schedule = linear(args.entropy_coef, args.entropy_coef * args.ent_final_frac)
    clip_schedule = linear(args.eps_clip, args.eps_clip * args.clip_final_frac)

    # Build PPO agent.
    sample_obs, _ = norm_envs[0].reset(seed=seed)
    state_dim = sample_obs.shape[0]
    action_dim = raw_envs[0].action_space.shape[0]
    agent = PPOAgent(
        state_dim, action_dim,
        lr=lr_schedule,
        gamma=args.gamma,
        eps_clip=clip_schedule,
        k_epochs=args.k_epochs,
        gae_lambda=args.gae_lambda,
        entropy_coef=ent_schedule,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        init_log_std=args.init_log_std,
        hidden_dim=args.hidden_dim,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
        value_clip=args.value_clip,
        state_dependent_std=args.state_dependent_std,
    )
    if args.load:
        print(f"[load]   {args.load}")
        payload = agent.load(args.load)
        if "obs_rms" in payload:
            canonical_rms.load_state_dict(payload["obs_rms"])

    # Optionally freeze the obs normaliser. With `--obs_norm_freeze_steps 0` the
    # running stats are frozen immediately on entry to the training loop — the
    # right choice when resuming from a checkpoint whose policy was tuned to the
    # *current* obs_rms snapshot. With a positive value, the normaliser updates
    # for that many env steps and then locks. -1 (default) means never freeze.
    if args.obs_norm_freeze_steps == 0:
        for w in norm_envs:
            w.training = False
        print("[obs_rms] frozen immediately (resumed-checkpoint mode)")

    # --- logging ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env}_{args.control}_{args.reward}_{timestamp}"
    print(f"[run]    {run_name}")
    csv_path = os.path.join(log_dir, f"training_log_{run_name}.csv")
    with open(csv_path, "w") as f:
        f.write("Update,EnvSteps,Episodes,Reward,Length,Difficulty,G,Friction,Threshold_Deg,"
                "PolicyLoss,ValueLoss,Entropy,KL,ClipFrac,EV\n")

    # Honour the harness's `EVOLVE_EVAL_DISABLE_TB=1` to skip the TB writer
    # for runtime measurements (TB filesystem chatter masks PPO timing).
    tb_disabled = os.environ.get("EVOLVE_EVAL_DISABLE_TB", "0") == "1"
    tb_dir = os.path.join(log_dir, "tb", run_name)
    writer = (SummaryWriter(tb_dir)
              if (SummaryWriter is not None and not tb_disabled)
              else None)
    if writer:
        print(f"[tb]     {tb_dir}")

    # --- curriculum state ---
    difficulty = float(args.start_difficulty)
    best_avg_reward = -float("inf")
    reward_window: collections.deque = collections.deque(maxlen=args.window)
    time_above_window: collections.deque = collections.deque(maxlen=args.window)
    _set_curriculum_all(venv, norm_envs, difficulty)
    updates_since_levelup = 0

    # Cache reward strategy reference for the time-above metric.
    ref_reward_strategy = raw_envs[0].reward_strategy
    max_theoretical_reward = _max_episode_reward(ref_reward_strategy, max_timesteps, raw_envs[0].dt)
    print(f"[max_R]  {max_theoretical_reward:.1f}")

    # Best-policy fallback state: track the highest deterministic-eval reward
    # we have seen, the policy snapshot at that moment, and a counter of
    # consecutive eval drops (used to roll the policy back to the best snapshot).
    best_eval_reward: float = -float("inf")
    best_eval_snapshot: dict | None = None
    best_eval_obs_rms: dict | None = None
    consecutive_eval_drops: int = 0

    # --- rollout buffers per env ---
    obs_arr = [norm_envs[i].reset(seed=seed + i)[0] for i in range(args.n_envs)]
    ep_rewards = [0.0] * args.n_envs
    ep_steps_above = [0] * args.n_envs
    ep_lengths = [0] * args.n_envs
    finished_rewards: list[float] = []
    finished_time_above: list[float] = []
    finished_lengths: list[int] = []
    total_episodes = 0
    total_env_steps = 0

    # --- Pre-allocated rollout buffers (candidate-3) -----------------------
    # The legacy code created per-env Python lists each update and ran
    # `np.asarray(traj_states[i])` to materialise them. With pre-allocated
    # (T, N, obs_dim) / (T, N, act_dim) numpy buffers we write directly into
    # contiguous memory and the post-rollout reshape is essentially free.
    n_envs_v = int(args.n_envs)
    T = int(rollout_steps)
    obs_dim_v = int(state_dim)
    act_dim_v = int(action_dim)
    buf_states = np.empty((T, n_envs_v, obs_dim_v), dtype=np.float32)
    buf_actions = np.empty((T, n_envs_v, act_dim_v), dtype=np.float32)
    buf_logp = np.empty((T, n_envs_v), dtype=np.float32)
    buf_rewards = np.empty((T, n_envs_v), dtype=np.float32)
    buf_dones = np.empty((T, n_envs_v), dtype=bool)
    # Per-step batched obs (input to the policy forward). Reused each step.
    obs_stacked = np.empty((n_envs_v, obs_dim_v), dtype=np.float32)
    # Pre-bind frequently-touched references to local names — avoids a
    # tower of attribute lookups in the inner loop.
    n_poles_v = raw_envs[0].n_poles
    reward_thresh_ref = raw_envs[0].reward_strategy
    csv_handle = open(csv_path, "a", buffering=1 << 16)

    try:
        for update in range(1, total_updates + 1):
            progress = (update - 1) / max(1, total_updates - 1)

            # ------------------------------------------------------------
            # Rollout (T steps, N parallel envs).
            #
            # Pre-allocated buffer layout (T, N, ...). Writing into a
            # contiguous numpy array per step avoids the per-env Python
            # list `.append` followed by `np.asarray(list)` materialise
            # that dominated the legacy path.
            # ------------------------------------------------------------
            for _t in range(T):
                # Stack current obs into the pre-allocated batch.
                for i in range(n_envs_v):
                    obs_stacked[i] = obs_arr[i]

                # Single batched policy forward across all envs.
                actions, log_probs = agent.select_action(obs_stacked, deterministic=False)
                if log_probs is None:
                    log_probs = np.zeros(n_envs_v, dtype=np.float32)

                buf_states[_t] = obs_stacked
                buf_actions[_t] = actions
                buf_logp[_t] = log_probs

                # Step each env. The Python loop is unavoidable (no vec-env
                # C extension here) but the batched policy forward above is
                # the dominant per-step cost — collapsing N forwards into 1
                # is the big win.
                for i in range(n_envs_v):
                    w = norm_envs[i]
                    next_obs, r, term, trunc, _ = w.step(actions[i])
                    done_flag = bool(term or trunc)
                    buf_rewards[_t, i] = r
                    buf_dones[_t, i] = done_flag

                    ep_rewards[i] += float(r)
                    ep_lengths[i] += 1
                    total_env_steps += 1

                    # Time-above-threshold accounting (uses raw env state, not normalised obs).
                    raw_state = raw_envs[i].state
                    above = True
                    for a in raw_state[1:1 + n_poles_v]:
                        # Same arctan2 wrap as the original — produces the
                        # same threshold check value.
                        err = abs(np.arctan2(np.sin(a - np.pi), np.cos(a - np.pi)))
                        if err >= reward_thresh_ref.reward_threshold:
                            above = False
                            break
                    if above:
                        ep_steps_above[i] += 1

                    if done_flag or ep_lengths[i] >= max_timesteps:
                        finished_rewards.append(ep_rewards[i])
                        finished_time_above.append(ep_steps_above[i] / max_timesteps)
                        finished_lengths.append(ep_lengths[i])
                        ep_rewards[i] = 0.0
                        ep_steps_above[i] = 0
                        ep_lengths[i] = 0
                        total_episodes += 1
                        next_obs, _ = w.reset()

                    obs_arr[i] = next_obs

                # Freeze the normaliser after a step-count warmup, if requested.
                if (args.obs_norm_freeze_steps > 0 and norm_envs[0].training
                        and total_env_steps >= args.obs_norm_freeze_steps):
                    for w in norm_envs:
                        w.training = False
                    print(f"[obs_rms] frozen at {total_env_steps} env steps")

            # ------------------------------------------------------------
            # GAE — vectorised across envs.
            #
            # The legacy path called `agent.compute_gae` once per env, each
            # call running its own critic forward (N total). We batch all
            # T*N states through the critic in ONE forward pass, then run
            # the recurrence per env on the resulting old_values matrix.
            # ------------------------------------------------------------
            # Flatten (T, N, obs_dim) -> (T*N, obs_dim) for one critic pass,
            # then reshape back to (T, N) for the per-env recurrence.
            with torch.no_grad():
                flat_states = buf_states.reshape(T * n_envs_v, obs_dim_v)
                flat_states_t = torch.as_tensor(flat_states, dtype=torch.float32, device=agent.device)
                flat_values = agent.policy.critic(flat_states_t).squeeze(-1).cpu().numpy().astype(np.float32)
                old_values = flat_values.reshape(T, n_envs_v)
                # Bootstrap values at the post-rollout state per env (for non-terminal tails).
                last_states = np.stack(obs_arr, axis=0).astype(np.float32)
                last_states_t = torch.as_tensor(last_states, dtype=torch.float32, device=agent.device)
                last_v = agent.policy.critic(last_states_t).squeeze(-1).cpu().numpy().astype(np.float32)

            # Per-env GAE recurrence (vectorised across envs at each timestep).
            # advantages[t, i] = delta[t, i] + gamma * lambda * (1 - done[t, i]) * advantages[t+1, i]
            advantages_v = np.zeros((T, n_envs_v), dtype=np.float32)
            gae_v = np.zeros(n_envs_v, dtype=np.float32)
            # Mask: 0 if terminal, 1 otherwise.
            masks = (~buf_dones).astype(np.float32)
            # Bootstrap: at the end of the rollout, next_value = last_v unless the
            # final step was terminal (in which case mask zeroes it out).
            next_value = last_v.copy()
            gamma_v = float(agent.gamma)
            lam_v = float(agent.gae_lambda)
            for t in range(T - 1, -1, -1):
                m_t = masks[t]
                delta = buf_rewards[t] + gamma_v * next_value * m_t - old_values[t]
                gae_v = delta + gamma_v * lam_v * m_t * gae_v
                advantages_v[t] = gae_v
                next_value = old_values[t]
            returns_v = advantages_v + old_values

            # Flatten and run the PPO update.
            states_cat = buf_states.reshape(T * n_envs_v, obs_dim_v)
            actions_cat = buf_actions.reshape(T * n_envs_v, act_dim_v)
            log_probs_cat = buf_logp.reshape(T * n_envs_v)
            adv_cat = advantages_v.reshape(T * n_envs_v)
            ret_cat = returns_v.reshape(T * n_envs_v)

            diag = agent.update_from_arrays(
                states_cat, actions_cat, log_probs_cat, adv_cat, ret_cat,
                progress=progress,
            )

            # --- metrics & ratchet ---
            for r in finished_rewards:
                reward_window.append(r)
            for t in finished_time_above:
                time_above_window.append(t)
            finished_rewards.clear()
            finished_time_above.clear()

            if len(reward_window) >= args.window:
                window_avg_reward = float(np.mean(reward_window))
                window_time_above = float(np.mean(time_above_window))

                previous_best = best_avg_reward
                if window_avg_reward > best_avg_reward:
                    best_avg_reward = window_avg_reward

                required_time_above = difficulty * 0.90
                if (
                    window_time_above > required_time_above
                    and window_avg_reward > previous_best
                    and difficulty < 1.0
                ):
                    # Adaptive ratchet step.
                    delta_R = window_avg_reward - max(previous_best, 0.0)
                    step = max(args.ratchet_min, min(
                        args.ratchet_max,
                        args.ratchet_max * (delta_R / max(1.0, max_theoretical_reward)),
                    ))
                    difficulty = min(difficulty + step, 1.0)
                    updates_since_levelup = 0
                    _set_curriculum_all(venv, norm_envs, difficulty)
                    print(
                        f"[ratchet] up={update:4d} d={difficulty:.3f} step={step:.3f} "
                        f"R={window_avg_reward:.1f} TimeAbove={window_time_above*100:.1f}%"
                    )
                else:
                    updates_since_levelup += 1

                if difficulty >= 1.0 and window_avg_reward > 0.95 * max_theoretical_reward:
                    print(f"[solved] update {update} R={window_avg_reward:.1f}")
                    final_path = os.path.join(log_dir, f"ppo_{run_name}_final.pth")
                    agent.save(final_path, extra={"obs_rms": canonical_rms.state_dict()})
                    return

            # --- CSV log ---
            avg_r = float(np.mean(reward_window)) if reward_window else 0.0
            avg_l = float(np.mean(finished_lengths)) if finished_lengths else 0.0
            avg_t = float(np.mean(time_above_window)) if time_above_window else 0.0
            threshold_deg = float(np.rad2deg(ref_reward_strategy.reward_threshold))
            # Buffered append to the persistent CSV handle. The legacy path
            # reopened the file every update (open + write + close); on
            # Windows this dominated wall-time when the per-update wall is
            # small (cold filesystem entries). One open at start of run +
            # buffered writes here is far cheaper.
            csv_handle.write(
                f"{update},{total_env_steps},{total_episodes},{avg_r:.4f},{avg_l:.2f},"
                f"{difficulty:.4f},{raw_envs[0].g:.3f},{raw_envs[0].friction_cart:.3f},"
                f"{threshold_deg:.2f},"
                f"{diag.get('policy_loss', 0):.4f},{diag.get('value_loss', 0):.4f},"
                f"{diag.get('entropy', 0):.4f},{diag.get('approx_kl', 0):.4f},"
                f"{diag.get('clip_fraction', 0):.4f},{diag.get('explained_variance', 0):.4f}\n"
            )

            if writer:
                writer.add_scalar("rollout/reward_mean", avg_r, total_env_steps)
                writer.add_scalar("rollout/length_mean", avg_l, total_env_steps)
                writer.add_scalar("rollout/time_above_mean", avg_t, total_env_steps)
                writer.add_scalar("curriculum/difficulty", difficulty, total_env_steps)
                writer.add_scalar("curriculum/g", raw_envs[0].g, total_env_steps)
                writer.add_scalar("curriculum/wind_std", raw_envs[0].wind_std, total_env_steps)
                for k, v in diag.items():
                    writer.add_scalar(f"ppo/{k}", v, total_env_steps)

            # --- console summary ---
            if update % args.log_interval == 0:
                print(
                    f"[upd {update:5d}] envsteps={total_env_steps:8d} "
                    f"ep={total_episodes:5d} d={difficulty:.2f} "
                    f"R={avg_r:7.1f} TimeUp={avg_t*100:5.1f}% "
                    f"KL={diag.get('approx_kl', 0):.3f} EV={diag.get('explained_variance', 0):+.2f}"
                )
                if update % args.save_interval == 0:
                    save_path = os.path.join(log_dir, f"ppo_{run_name}_{update}.pth")
                    agent.save(save_path, extra={"obs_rms": canonical_rms.state_dict()})

            # --- evaluation ---
            if args.eval_every > 0 and update % args.eval_every == 0:
                eval_metrics = _evaluate(args, agent, canonical_rms.state_dict(),
                                         difficulty, args.eval_episodes, max_timesteps,
                                         seed=seed + 1_000_000)
                print(f"[eval]   {eval_metrics}")
                if writer:
                    for k, v in eval_metrics.items():
                        writer.add_scalar(k, v, total_env_steps)

                # Best-policy fallback. Snapshot whenever eval improves; roll
                # back if eval drops below `best_fallback_threshold` of the
                # best for `best_fallback_patience` consecutive evaluations.
                cur_eval = eval_metrics["eval/reward_mean"]
                if cur_eval > best_eval_reward:
                    best_eval_reward = cur_eval
                    best_eval_snapshot = {k: v.detach().cpu().clone()
                                          for k, v in agent.policy.state_dict().items()}
                    best_eval_obs_rms = canonical_rms.state_dict()
                    consecutive_eval_drops = 0
                    if args.best_fallback_save:
                        path = os.path.join(log_dir, f"ppo_{run_name}_best.pth")
                        agent.save(path, extra={"obs_rms": canonical_rms.state_dict()})
                elif (args.best_fallback_threshold > 0.0
                      and best_eval_snapshot is not None
                      and cur_eval < args.best_fallback_threshold * best_eval_reward):
                    consecutive_eval_drops += 1
                    if consecutive_eval_drops >= args.best_fallback_patience:
                        print(f"[fallback] eval dropped {consecutive_eval_drops}x below "
                              f"{args.best_fallback_threshold:.2f} * best ({best_eval_reward:.1f}); "
                              f"rolling back policy to best snapshot.")
                        agent.policy.load_state_dict(best_eval_snapshot)
                        if best_eval_obs_rms is not None:
                            canonical_rms.load_state_dict(best_eval_obs_rms)
                        consecutive_eval_drops = 0
                else:
                    consecutive_eval_drops = 0

    except KeyboardInterrupt:
        print("[interrupt] saving checkpoint and exiting.")
    finally:
        # Flush and close the buffered CSV handle before saving the final
        # checkpoint, so all pending updates are visible on disk if the
        # process is interrupted later in this finally block.
        try:
            csv_handle.flush()
            csv_handle.close()
        except Exception:
            pass
        save_path = os.path.join(log_dir, f"ppo_{run_name}_final.pth")
        agent.save(save_path, extra={"obs_rms": canonical_rms.state_dict()})
        print(f"[done]   final checkpoint: {save_path}")
        if writer:
            writer.close()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Env / strategies.
    p.add_argument("--env", type=str, default="double", choices=["single", "double"])
    p.add_argument("--control", type=str, default="velocity", choices=["force", "velocity"])
    p.add_argument("--reward", type=str, default="exponential",
                   choices=["standard", "exponential", "energy", "lqr", "hybrid"])
    p.add_argument("--reset_mode", type=str, default="down")
    p.add_argument("--integrator", type=str, default="rk4", choices=["rk4", "semi_implicit"])
    p.add_argument("--x_soft", type=float, default=3.5)
    p.add_argument("--x_hard", type=float, default=10.0)
    p.add_argument("--boundary_penalty_k", type=float, default=0.1)
    p.add_argument("--wind_max", type=float, default=1.0)
    p.add_argument("--survival_alpha", type=float, default=0.0,
                   help="Smooth survival bonus added to standard rewards.")
    p.add_argument("--threshold_curve", type=str, default="linear",
                   choices=["linear", "concave"],
                   help="Schedule for the upright-tolerance threshold epsilon(delta).")
    p.add_argument("--obs_norm_freeze_steps", type=int, default=-1,
                   help="-1: never freeze. 0: freeze immediately (resume mode). "
                        ">0: freeze after that many env steps of warmup.")

    # Training scale.
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--rollout_steps", type=int, default=512)
    p.add_argument("--episode_steps", type=int, default=4000)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=None)

    # PPO hyperparameters.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_final_frac", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--clip_final_frac", type=float, default=0.5)
    p.add_argument("--k_epochs", type=int, default=10)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--ent_final_frac", type=float, default=0.1)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--init_log_std", type=float, default=-0.5)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--minibatch_size", type=int, default=256)
    p.add_argument("--target_kl", type=float, default=0.03)
    p.add_argument("--value_clip", type=float, default=None,
                   help="If set, enables PPO2-style clipped value loss with this half-width.")
    p.add_argument("--state_dependent_std", action="store_true",
                   help="Make the policy log-std a state-dependent head (allows "
                        "shrinking noise near upright; necessary for tight LQR-style "
                        "stabilisation). Defaults to off (state-independent learnable scalar).")

    # Curriculum.
    p.add_argument("--start_difficulty", type=float, default=0.0)
    p.add_argument("--ratchet_min", type=float, default=0.005)
    p.add_argument("--ratchet_max", type=float, default=0.05)
    p.add_argument("--window", type=int, default=20,
                   help="Number of completed episodes for ratchet-window stats.")

    # Eval / log.
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=50,
                   help="Run a deterministic-policy eval every N PPO updates (0 disables).")
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--best_fallback_threshold", type=float, default=0.0,
                   help="If >0, when eval reward drops below this fraction of the "
                        "all-time best for `best_fallback_patience` consecutive "
                        "evaluations, roll the policy back to the best snapshot. "
                        "0 disables the fallback.")
    p.add_argument("--best_fallback_patience", type=int, default=3)
    p.add_argument("--best_fallback_save", action="store_true",
                   help="Save the running best-eval snapshot to ppo_<run>_best.pth.")

    p.add_argument("--load", type=str, default=None)
    return p


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    train(args)
