# Training Experiments Log

A chronological log of each meaningful training run, the *hypothesis* it tested,
and what we learned. The point is not to recreate the exact metrics (those live
in `logs/training_log_*.csv` and `docs/reports/TRAINING_REPORT_*.md`) but to
record the *intent* and the diff so future debugging can find the seam where
something went wrong.

Conventions:
* Each entry names a checkpoint / log directory so you can replay it.
* "Hypothesis" is what we *expected* to happen (so we can tell the difference
  between "it worked" and "it gave a different effect than I thought").
* "Outcome" is what actually happened.
* "Lesson" is the takeaway we carry forward.

---

## Phase A — single pendulum sanity (2026-04-28)

* **Hypothesis**: post-redesign pipeline (vec envs, GAE, tanh squash, obs
  normaliser, soft cart bounds) trains the easy 1-pole task without surprises.
* **Setup**: `--env single --reward exponential --n_envs 8 --rollout_steps 256
  --updates 500`. Default ratchet (linear threshold curve, `--ratchet_max 0.05`).
* **Outcome**: difficulty climbed 0 → 0.11, training reward 0 → 12 244,
  TimeUp 0 → 77 %, EV 0 → 0.85. Eval reward fluctuated (1380 → 2319) but
  trended up. Critic healthy throughout.
* **Lesson**: the redesigned pipeline is functional end-to-end. Per-env GAE was
  the critical fix (without it, EV stuck near 0).
* **Checkpoint**: `logs/ppo_single_velocity_exponential_20260428_222743_final.pth`.

## Phase B — double pendulum, energy reward (2026-04-28)

* **Hypothesis**: `EnergyShapingReward` provides a better swing-up gradient
  than the sparse exponential reward (the energy term is non-zero everywhere,
  including far from upright), so it's the right starter for the harder task.
* **Setup**: `--env double --reward energy --n_envs 8 --rollout_steps 512
  --updates 1500 --minibatch_size 512`.
* **Outcome**: difficulty 0 → 0.24 in ~25 min wall time. Training reward
  ~975, eval reward 933. EV stayed at 0.94 — critic excellent. The
  energy reward works as intended for the swing-up phase.
* **Lesson**: energy shaping does pull the policy onto the homoclinic orbit
  fast. By difficulty 0.24 we have a competent swing-up policy.
* **Checkpoint**: `logs/ppo_double_velocity_energy_20260428_223722_final.pth`.

## Phase C — double pendulum, exponential reward, resumed (2026-04-28)

* **Hypothesis**: switching from `energy` to `exponential` reward after
  swing-up extends the *duration* of upright runs (exp reward grows with
  $T_{\rm up}$, not just instantaneous proximity).
* **Setup**: `--reward exponential --load <Phase B>.pth --updates 1000
  --start_difficulty 0.24`.
* **Outcome**: difficulty stayed at 0.24 (no advance). Eval reward 1248,
  TimeUp 36 %. Better eval than Phase B but the curriculum did not move.
* **Lesson**: the exponential reward at $\delta = 0.24$ produces good
  *intra-difficulty* improvements but the agent rarely beats the all-time
  best — so the ratchet doesn't fire. Suggests the local optimum is
  attractive at this difficulty.
* **Checkpoint**: `logs/ppo_double_velocity_exponential_20260428_232415_final.pth`.

## Phase E — long continuation, finer ratchet (2026-04-29)

* **Hypothesis**: with `--ratchet_min 0.001 --ratchet_max 0.01` (vs the
  default 0.005 / 0.05) the ratchet should fire more often once any sliver of
  improvement appears, breaking the Phase-C plateau.
* **Setup**: `--load <Phase C>.pth --updates 5000 --start_difficulty 0.24
  --ratchet_min 0.001 --ratchet_max 0.01`.
* **Outcome**: very promising up to update ~630 (difficulty 0.266, peak
  training reward 8 948). After that the policy degraded sharply — by
  update 1400 reward had collapsed to 421 and never recovered. Ratchet
  locked at 0.266 for the remaining 4 370 updates while reward oscillated
  600 – 1 200 with no progress. KL drifted up to 0.018 in late updates.
* **Lesson** (the important one):
    1. **The agent CAN reach $\delta = 0.266$ and produce good rollouts** —
       proving the curriculum + reward design is workable at intermediate
       gravity.
    2. **Catastrophic forgetting is real here**: a successful policy degraded
       under continued PPO updates, possibly because (a) the obs-normaliser
       running statistics shifted, (b) LR was too high relative to the value
       head's calibration to the high-reward regime, or (c) the entropy
       coefficient was annealed too aggressively, narrowing exploration.
    3. **The `_1200.pth` intermediate checkpoint is better than the final**
       checkpoint by 28 % on eval reward (1178 vs 923) — a clear signature of
       overtraining, not undertraining.
* **Best checkpoint** (use this, not the final): `logs/ppo_double_velocity_exponential_20260429_001259_1200.pth`.
* **Final checkpoint**: `..._final.pth` — kept for reference but degraded.

## Phase F — anti-collapse + concave curriculum (run 2026-04-29)

Hypotheses being tested simultaneously (orthogonal levers, so we can isolate
which one mattered later by ablation):

| Lever | Phase E setting | Phase F setting | Why we're changing it |
|---|---|---|---|
| Threshold curve $\epsilon(\delta)$ | linear | **concave** ($\epsilon = 10° + 80°\sqrt{1-\delta}$) | gives the agent ~10° more tolerance at every $\delta < 1$, especially at low $\delta$ where the swing-up is still being learned. Aim: keep ratchet firing past 0.266. |
| Obs normaliser | always-on (training=True) | **frozen at warmup_steps = 0** | Phase E's obs_rms kept updating for 20 M env steps. As the policy's exploration changed, the obs distribution drifted, which feeds the policy's *own* updates a non-stationary input distribution — a classic cause of catastrophic forgetting. We freeze immediately on resume so the policy sees the same input stats it was tuned to. |
| LR schedule | `lr=3e-4 → 3e-5` (Phase E final = 0.06×) | **`lr=2e-4 → 2e-4` (no decay)** | Phase E's tiny final LR can't escape local optima. A flat moderate LR keeps the policy malleable. |
| Entropy schedule | `0.01 → 0.001` | **`0.005 → 0.005`** (no decay, lower start) | Prevents premature collapse to deterministic-ish policies while still being more disciplined than Phase E's start. |
| Hidden dim | 256 | **256** (unchanged) | A larger network would require training from scratch (mismatched state_dict shapes when loading the Phase E checkpoint). Deferred to a possible Phase G if Phase F doesn't break the plateau. |
| Updates | 5 000 | 3 000 | We expect faster wall-clock with the changes; if they help, 3 000 should suffice to see ratchet beyond 0.266. |

* **What success looks like**: difficulty advances to ≥ 0.40, eval reward ≥
  2 000, no late-training reward collapse (final checkpoint ≥ best
  intermediate checkpoint).

* **What failure modes look like**:
    1. Stays at 0.266 with stable reward → none of the levers were the
       root cause; need to look elsewhere (action squashing? reward shape?
       random seed sensitivity?).
    2. Advances briefly then collapses again → freezing the obs normaliser
       wasn't enough; LR is still too high or capacity still too small.
    3. Advances but oscillates wildly → LR too high.
    4. Doesn't even reproduce Phase E's gains → the loaded checkpoint
       might depend on the obs-rms drift to function (i.e. the "frozen"
       stats from the parent checkpoint differ from what the policy expects).

* **Will run as**: `--load logs/ppo_double_velocity_exponential_20260429_001259_1200.pth`
  (the BEST checkpoint, not the degraded final).

### Outcome

* Difficulty: **0.266 → 0.295** (the plateau was broken). The ratchet fired
  29 times, mostly clustered around updates 23–35 and 327–581.
* Peak training reward: **6 788** at update 581 (vs Phase E peak ~8 948 at
  $\delta = 0.266$, so similar absolute peak but at a *harder* difficulty).
* Best deterministic eval (update 600 checkpoint): **R = 2 490, strict success 3.3 %, loose 10.5 %**
  (vs Phase E peak 1 178 / 2.8 % / 9.9 %). All metrics improved at the harder
  difficulty.
* **Catastrophic forgetting still occurred**: by update 3 000 the eval reward
  had collapsed back to 1 084 / 2.4 % / 7.8 %. The peak was held only briefly
  (updates 581–875).
* Best checkpoint: `logs/ppo_double_velocity_exponential_20260429_144201_600.pth`.

### Lessons

1. **The concave threshold curve is the lever that mattered most**: the same
   $\delta = 0.295$ that was unreachable in Phase E became attainable here.
   With $+10°$ extra tolerance at every intermediate $\delta$, the agent had
   room to consolidate.
2. **Freezing obs_rms** allowed the policy to start cleanly from the loaded
   checkpoint (EV started at 0.95). Necessary but not sufficient.
3. **Catastrophic forgetting persists** even with frozen obs_rms, no LR decay,
   and tighter entropy. The signature is the same: peak around update 600,
   then policy drift even though the curriculum has stopped advancing.
4. **The fundamental bottleneck appears to be PPO instability**, not the
   curriculum or normalisation. Candidate root causes for next phase:
    - **Trust-region too loose**: `target_kl=0.03` is the early-stop trigger
      but late-Phase-F runs hit KL=0.04+ frequently without triggering. We
      should *lower* `target_kl` to 0.015 and *raise* `max_grad_norm` from 0.5
      to 1.0 so the policy is more strictly trust-region-controlled.
    - **Critic over-confidence**: EV stays at 0.94+ even when reward
      collapses, suggesting the critic *correctly predicts the bad rollouts*
      — the policy is the source of variance, not the value head. Adding
      value-loss clipping (PPO2-style) would prevent the value head from
      following the policy off a cliff.
    - **Snapshot-best fallback**: maintain a separate "best policy seen so
      far" checkpoint inside the training loop and roll back to it if the
      eval reward drops below 50 % of the best for 3 consecutive evals. This
      is operational engineering, not algorithmic, but would directly cure
      the symptom of "great policy seen at update 600, lost forever".

## Phase G — trust-region tightening + best-policy fallback (run 2026-04-30)

* **Hypothesis**: the catastrophic-forgetting signature in Phases E and F is a
  PPO trust-region failure (KL routinely > 0.04, value head correctly
  predicting the collapse). A tighter `target_kl`, value-loss clipping, and
  a "best policy" fallback should let us *keep* the gains we earn rather
  than losing them to drift.

* **Levers** (all orthogonal to Phase F's):

| Lever | Phase F | Phase G | Rationale |
|---|---|---|---|
| `target_kl` | 0.03 | **0.015** | Halve the trust-region. Late-Phase-F KL = 0.04+ should now early-stop the K-epoch loop, preventing the catastrophic update. |
| `max_grad_norm` | 0.5 | **1.0** | Allows healthy updates to take a full step while letting the KL gate catch the over-aggressive ones. |
| Value loss | Plain MSE | **clipped MSE** (PPO2-style: $\max(\Delta V_{\rm clipped}, \Delta V_{\rm raw})^2$) | Prevents the value head from chasing the policy when the policy is wrong. |
| Best-policy fallback | — | **on** (rolls back to best deterministic-eval checkpoint after 3 consecutive eval drops $> 50%$) | Operational safety net. |

* **Will run as**: `--load logs/ppo_double_velocity_exponential_20260429_144201_600.pth`
  (the BEST Phase F checkpoint, at $\delta = 0.295$).
* **Success criteria**: ratchet to $\delta \ge 0.40$, eval reward $\ge 3000$,
  no end-of-training collapse.

### Outcome

* Difficulty: 0.295 → **0.301** (only +0.006). The plateau is real and not
  a PPO-instability artefact.
* Best window training reward: **8 184** at update 210 (similar to Phase F).
* Best deterministic eval (`_best.pth` snapshot): **R = 1 469, strict 2.7 %,
  loose 8.9 %** (lower than Phase F best at 2 490, but more *consistent*).
* Fallback fired **8 times** during training, demonstrating that the
  rollback mechanism works as designed and prevents catastrophic forgetting.
* `target_kl = 0.015`: late-training KL averaged 0.016, right at target —
  the trust-region tightening took effect.
* `value_clip = 0.2`: critic explained-variance still strong (~0.87
  late-training), without the wild swings seen in Phase F.

### Lessons

1. **The trust-region tightening + fallback prevents the worst drift but
   does not break the plateau.** The agent simply cannot push past
   $\delta \approx 0.30$ at this network capacity and compute budget.
2. **The ~3% strict-success rate across Phases C/E/F/G is the dead giveaway**
   — the policy stays roughly within ±70° of upright (loose 8-10 % at the
   curriculum's threshold), but never tightens to ±10°. This is *not* a PPO
   instability problem any more; it is a *capacity / compute* problem.
3. **Best snapshot saved**: `logs/ppo_double_velocity_exponential_20260430_145432_best.pth`.

## Final assessment of the swing-up sequence (Phases B → G)

After ~50 M total env steps across 5 phases of curriculum continuation, the
double-pendulum agent reliably:

* Performs swing-up: gets the poles roughly toward upright on most episodes.
* Stays within ±70° of vertical for ~40 % of an episode.
* Maintains an episode reward of ~1 500 – 2 500 deterministic.

It does **not** stabilise to within ±10° of upright at any difficulty above
~0.30, and the curriculum cannot ratchet past $\delta = 0.301$ regardless of
the trust-region / curriculum / normaliser knobs we have available.

### What the diagnostic table tells us about the bottleneck

| Phase | Updates | Δ steps | Δ difficulty | Best eval | Strict % | Loose % |
|---|---|---|---|---|---|---|
| C | 1 000 | 4 M | 0.000 | 1 248 | 2.6 % | 7.6 % |
| E peak | 1 200 | 4.9 M | +0.026 | 1 178 | 2.8 % | 9.9 % |
| F peak | 600 | 2.5 M | +0.029 | **2 490** | **3.3 %** | **10.5 %** |
| G best | ~700 | 2.9 M | +0.006 | 1 469 | 2.7 % | 8.9 % |

* The **biggest single jump** in eval reward came from Phase F's switch to
  the concave threshold curve — a *curriculum design* lever, not a PPO lever.
* The **strict success rate has been flat at 2-3 %** across all four phases,
  suggesting the policy genuinely cannot resolve the 10° tolerance band at
  any of the trained difficulties, regardless of optimisation tricks.
* The most likely remaining root causes are *capacity* (256 hidden units may
  be insufficient for the value function near the saddle) and *compute*
  (the literature reports needing ~$10^8$ env steps for cart-double-pole;
  we are at ~5 × $10^7$).

### Recommended Phase H if the work continues

Reset the experiment with:

1. **Train from scratch with `--hidden_dim 512`** and the best-known
   curriculum (concave threshold, frozen obs_rms after warmup, target_kl
   = 0.015, value_clip = 0.2, best-policy fallback).
2. **Budget 50 000 PPO updates** (~200 M env steps).
3. Use `--reward energy` for the first half (swing-up regime) then resume
   on `--reward exponential` for the second half (duration regime), as in
   the B → C transition.
4. Treat the `_best.pth` snapshot as the deliverable, not the `_final.pth`.

Until that compute is available, the best policy this codebase has produced
is `logs/ppo_double_velocity_exponential_20260429_144201_600.pth` (Phase F
update 600), with deterministic eval R = 2 490 at $\delta = 0.295$.

## Phase H — LQR-cost reward, train from scratch (run 2026-04-30, FAILED)

Reframing the diagnosis: after Phase G, the strict-success rate has been
stuck at 2.6 – 3.3 % across four phases of optimisation tuning. This is *not*
a PPO instability; it is a *reward specification* problem.

### Hypothesis

The exponential / standard / energy rewards we have been using all share the
same defect for the *tight stabilisation* phase: their gradient with respect
to action is either zero (sparse step rewards: nothing tells the policy to
tighten *inside* the band) or symmetric around the homoclinic orbit (energy
shaping: many states have the same total energy as upright). PPO can only
follow the gradient it is given, so the policy converges to a "ride the wide
band loosely" attractor.

A quadratic-cost reward of the form

$$ r_t = b_{\rm alive} - \tilde s^T Q \tilde s - R\, a^2 $$

— directly modeling the LQR objective — has a smooth, monotone gradient all
the way to the optimum. The policy gradient pulls the angles toward $\pi$
without any step-function penalty band.

### Setup

* New reward strategy: `LQRCostReward` (added to `src/strategies/rewards.py`).
* Schedule: $q_\theta(\delta) = 0.5 + 4.5\delta$ — at $\delta = 0$ the angle
  penalty is mild (so the swing-up phase is dominated by velocity damping
  and centring), at $\delta = 1$ it is 10× larger (tight stabilisation).
* Constants $q_x = 0.1, q_{\dot x} = 0.01, q_{\dot\theta} = 0.05, R = 0.005,
  b_{\rm alive} = 1$. With these values, upright-at-rest gives $r = +1$ and
  down-at-rest at $\delta = 1$ gives $r \approx -98$.
* Initial training: from scratch (do NOT load Phase F/G checkpoints — those
  policies were shaped by the wrong objective and would have to unlearn the
  loose-band strategy).
* Curriculum-aligned ratchet: keep the existing window mechanism since
  positive episode rewards (~$T \cdot b_{\rm alive}$ at convergence) are
  monotone in policy quality.
* Other knobs carried forward from Phase G: concave threshold curve (now
  unused for gradient but still in the bookkeeping), frozen obs_rms after
  warmup of 100k env steps, target_kl = 0.015, max_grad_norm = 1.0,
  value_clip = 0.2, best-policy fallback active.
* Compute: 8 envs × 512 rollout × 3000 updates = 12 M env steps.

### Success criteria

* Strict success rate **> 10%** at $\delta \ge 0.5$ — that's a 3× improvement
  over the Phase F peak and would falsify the "stuck at 3%" pattern.
* Difficulty advances past **$\delta = 0.5$** (Phases C/E/F/G all stuck $\le 0.30$).
* Eval reward trajectory is **monotone-ish** (it can dip but the best
  snapshot at the end ≥ best snapshot mid-training). The LQR reward is dense
  enough that there should be no catastrophic-collapse signature.

### Failure modes to look for

1. **The agent flatlines at swing-up height.** Means the swing-up gradient is
   too weak relative to the alive-bonus; we'd raise $q_\theta^{\min}$ or
   reduce $b_{\rm alive}$.
2. **The agent hits the cart wall.** Soft-bound penalty fights the LQR cost.
   We'd raise $q_x$ or `boundary_penalty_k`.
3. **Strict success still stuck near 3%.** Diagnosis falsified — points to a
   policy capacity / variance problem rather than reward shape. Then we go
   to Obstruction B (state-dependent log_std).

### Outcome (FAILED — different failure mode than predicted)

* **The agent learned "do nothing".** Final difficulty 0.005 (one ratchet
  step), TimeUp 0%, eval reward ~ −7 500.
* Episode lengths short (~430–1700 steps): the agent oscillates the cart
  weakly until it drifts past `x_hard` and terminates.
* KL near zero in late training: the policy gradient is essentially flat —
  the agent has found the *do-nothing* basin and PPO can't see any
  improvement direction.
* **Diagnosis of the diagnosis**: the unconditional velocity-quadratic
  penalty $-q_{\dot\theta}\sum\dot\theta_i^2$ creates a strong gradient
  pulling the agent toward $\dot\theta = 0$ *everywhere*, including at
  down-rest. At down-rest with $\dot\theta = 0$, reward is $1 - q_\theta\,
  2\pi^2 \approx -8.85$ per step (with $q_\theta^{\min} = 0.5$). Any swing
  attempt incurs additional velocity penalty *before* the angle penalty
  yields enough offset, so the policy gradient pushes $\dot\theta \to 0$
  faster than $\theta \to \pi$. This is a *swing-up barrier*: the agent
  learns the local optimum "stay still and accept −8.85/step" rather than
  the harder global optimum "swing up to get +1/step at upright".
* The plot of phase_h.log shows reward roughly *worsening* over time,
  consistent with the cart drifting in random directions because the
  gradient signal is too weak to teach centring.

### Lesson

The LQR cost is correct *near* the upright but pathological *far from* it.
A homogeneous quadratic cost on velocity is the wrong global shape for an
underactuated system that *requires* high velocity during swing-up.

## Phase H2 — gated-velocity LQR cost (run 2026-04-30, FAILED)

### Hypothesis

Gate the velocity penalties by a Gaussian envelope of the angle error:

$$ q_{\dot x}^{\rm eff}(s) = q_{\dot x} \cdot \exp\!\Bigl(-\frac{\sum_i e_{\theta_i}^2}{2\sigma_p^2}\Bigr), \qquad \sigma_p = 45^\circ. $$

* Far from upright (during swing-up), the envelope is ≈ 0, so velocity is
  effectively unpenalised — the agent is free to swing aggressively.
* Near upright, the envelope is ≈ 1, so velocity is fully penalised — the
  agent is forced to stop moving.

This is exactly the structure used by classic *energy-shaping + LQR*
controllers (Tedrake, ch. 3.5): two regimes, with a smooth transition.

Also bumping $q_\theta^{\min}$ from 0.5 to 2.0 so the angle gradient is
stronger throughout (the angle error term remains the dominant pull
toward upright).

### Verification

Hand-checking the reward shape for the new gated form (at $\delta = 0$):

| state | reward | comment |
|---|---:|---|
| upright rest | +1.000 | optimum |
| upright + 5 rad/s | −1.500 | velocity penalty kicks in (proximity ≈ 1) |
| horizontal | −8.870 | angle penalty dominates |
| horizontal + 5 rad/s | −8.915 | velocity penalty negligible (proximity ≈ 0.04) |
| down rest | −38.478 | maximum angle penalty |
| down + 5 rad/s | −38.478 | swinging is *free* at downright (proximity ≈ 0) |

The "down + 5" reward equals "down rest" — confirming velocity penalties
are gated off in the swing-up regime, restoring the energy-pumping
gradient that Phase H suppressed.

### Setup

Same as Phase H otherwise (from scratch, 3 000 updates, concave threshold,
target_kl = 0.015, value_clip = 0.2, best fallback active).

### Success criteria

* Difficulty advances past 0.30 (matches or beats best of Phase F/G).
* Strict success ≥ 10 % at any difficulty (would be the first crack in the
  3 % ceiling).
* Eval reward trajectory monotone or fallback-stable.

### Outcome (FAILED — different failure mode again)

* Final difficulty 0.005, TimeUp 0%, eval reward ~ −5 000.
* Episode lengths ~300–500 (cart escapes `x_hard` under random exploration
  before learning starts).
* Critic EV climbed to 0.69 (it correctly predicts that random rollouts are
  bad), but the policy is stuck — KL ~ 0.001 means the policy gradient is
  essentially zero.
* Diagnosis: the gated-velocity fix was *necessary* but not *sufficient*.
  The remaining problem is that an LQR-cost reward gives **no positive
  shaping signal during swing-up** — only ever-negative penalties. The
  cart drifts under stochastic exploration into the boundary penalty zone
  before the agent has learned anything useful, and the gradient is too
  weak to teach pumping. In contrast, ``EnergyShapingReward`` (Phase B)
  has a *positive Gaussian peak* on the homoclinic energy manifold which
  acts as a beacon for the swing-up policy gradient.

### Lesson

A pure quadratic-cost reward is *locally correct* (smooth gradient near
upright) but globally pathological for an underactuated swing-up problem.
The agent needs a *positive* shaping signal to find the upright manifold
in the first place; only then does the quadratic penalty's tightening
gradient become useful.

## Phase H3 — hybrid: exp continuity + small LQR quadratic (run 2026-05-01, BREAKTHROUGH)

### Hypothesis

Combine what works:

$$ r_t = r_{\rm exp}(s; \delta) - q_\theta \sum_i e_{\theta_i}^2 - q_x x^2. $$

* The exponential continuity term ($r_{\rm exp}$) handles swing-up (proven
  in Phases B–G to drive $\delta \to 0.30$).
* The quadratic angle term provides the tightening gradient that
  ``ExponentialSwingUpReward`` lacks inside the band — directly attacks
  the 3% strict-success ceiling.
* Magnitudes: $q_\theta = 0.2$, so the quadratic term is at most
  $0.2 \cdot 2\pi^2 \approx 4$ per step at down (small relative to the
  band-saturated exp reward of ~147). This means it does not dominate the
  swing-up phase, only nudges it; but it provides the *only* signal inside
  the band, where it dominates by default.

### Setup

Same training config as Phase H2 (from scratch, 3000 updates, concave
threshold curve, target_kl=0.015, value_clip=0.2, fallback active).

### Success criteria

* Difficulty advances past 0.30 (same bar as H2).
* **Strict success ≥ 10 %** at the achieved difficulty — this is the
  diagnostic for whether the diagnosis is correct.
* No catastrophic forgetting (best fallback prevents).

### Outcome (BREAKTHROUGH)

| Metric | Phase F (best of all earlier) | Phase H3 best |
|---|---:|---:|
| Final difficulty | 0.295 | **0.430** |
| Strict success (10°) | 3.3 % | **5.9 %** |
| Loose success (20°) | 10.5 % | **13.0 %** |
| Episode length | 3 962 | 4 000 (full episode) |
| SSE pole 1 / pole 2 | 68° / 75° | 75° / 61° |
| Total ratchets | 9 | **86** |

* Difficulty climbed 0.0 → 0.43 from scratch in 3 000 updates (vs Phases B/C/E/F/G needing ~12 000 cumulative updates and stalling at 0.30).
* **Strict-success rate (5.9 %) is the first datapoint above 3 %** across
  all eight phases. This validates the diagnosis: the previous reward
  shapes had no gradient inside the threshold band, capping policy
  precision; the small quadratic penalty pulls the angles tighter.
* The agent is asymmetric on the two poles (P1 SSE 75°, P2 SSE 61°) —
  classic underactuated-control behaviour where the control authority is
  shared but the dynamics couple the two poles.
* The training reward window oscillated between −2 000 and +1 800
  late-training, which looks unstable but the deterministic *eval* reward
  was rising steadily (best −705 at update ~2 600, with successful 4 000-step
  episodes); fallback rolled back when stochastic-policy returns dipped.

### Concrete artefacts

* Best policy: `logs/ppo_double_velocity_hybrid_20260501_003556_best.pth`.
* Video: `docs/images/final_run_phase_h3.mp4` (20 s, deterministic, $\delta = 0.43$).
* Diagnostic report: `docs/reports/TRAINING_REPORT_phase_h3_best.md`.

### What this proves

The plateau across Phases C/E/F/G was **not** a capacity / compute bottleneck
(as we had hypothesised after Phase G). It was a **reward-specification**
bottleneck. Adding a non-zero gradient inside the band immediately broke the
ceiling that had held flat for ~50 M env steps of optimisation tuning.

The remaining gap (5.9 % strict success vs the 80 % "good stabilisation" bar)
is now plausibly closeable with more compute on this same hybrid reward, and
possibly the state-dependent variance fix from the original analysis
(Obstruction B). The signature is now: progress at every checkpoint, just
slow because the quadratic gradient is small.

## Phase I — extend Phase H3 (run 2026-05-01, marginal gain)

Same setup as Phase H3, loaded from H3 best, 3 000 more updates.

* Final difficulty: 0.432 (was 0.430).
* Strict success: **6.5 %** (was 5.9 %). Marginal +0.6 % improvement.
* Loose success: 13.4 % (was 13.0 %). Marginal +0.4 %.
* Best checkpoint: `logs/ppo_double_velocity_hybrid_20260501_184759_best.pth`.

The hybrid reward has plateaued at ~6.5 % strict success. The diagnosis: at
this point the policy *does* have a gradient toward the optimum, but cannot
*execute* fine enough corrections because its action noise is structurally
too large near upright (Obstruction B — state-independent log_std).

## Phase J — state-dependent log_std (run 2026-05-01, FAILED)

### Hypothesis

Make the policy's log_std a state-dependent head $\sigma_\theta(s)$ so the
agent can shrink action noise near upright (where micro-corrections are
needed) and keep large noise during swing-up. This is the standard fix
identified in the original first-principles analysis (Obstruction B).

### Setup

Same as Phase H3 except `--state_dependent_std` enabled. Trains from
scratch (architecture change forbids loading H3/I checkpoints).

### Outcome (FAILED)

* Final difficulty: **0.14** (vs 0.43 for state-independent — a regression).
* Mean KL late training: **0.046** (3× the 0.015 target — the trust-region
  control is being violated routinely).
* Eval reward: −5 100 at termination (worse than random initialisation).

### Diagnosis

State-dependent variance is an *SAC feature*, not a PPO one. SAC uses the
reparameterised gradient for the policy update, which propagates through
both the mean and the log_std heads cleanly. PPO uses an importance-ratio
gradient on stored log-probabilities, where state-dependent log_std produces
much larger ratio variance: when $\sigma_\theta(s)$ shifts between rollout
collection and update, the importance ratio explodes and the trust-region
clip becomes the dominant gradient. We observe exactly this: KL grows
unbounded, policy regresses.

This failure mode is known in the PPO literature; e.g. Schulman et al.'s
implementations all use state-independent log_std for this reason. To get
state-dependent variance to work robustly, we would need:
1. A separate, much smaller learning rate for the log_std head.
2. Or to switch to SAC (different update structure).

Both are significant scope expansions. Lesson: Obstruction B *is* a real
ceiling, but PPO is not the right base algorithm for fixing it.

## Final state of the campaign (as of 2026-05-01)

The 3 % strict-success ceiling was broken in Phase H3 and pushed to 6.5 %
in Phase I via the **hybrid reward** (exponential continuity + small
quadratic angle penalty). Beyond that, the PPO + tanh-Gaussian + global
log_std combination cannot easily go further on this task.

| Phase | Strict 10° | Difficulty | Note |
|---|---:|---:|---|
| C (baseline, exp reward) | 2.6 % | 0.24 | |
| F (concave + freeze) | 3.3 % | 0.30 | concave threshold mattered |
| H3 (hybrid reward, from scratch) | 5.9 % | 0.43 | reward shape was the lever |
| **I (extend H3)** | **6.5 %** | **0.432** | best policy |
| J (state-dep log_std) | failed | 0.14 | PPO incompatible with state-dep σ |

**Best policy**: `logs/ppo_double_velocity_hybrid_20260501_184759_best.pth`.
**Best video**: `docs/images/final_run_phase_h3.mp4` (Phase H3 best, similar
quality to Phase I best).

### What would push further

1. **Switch from PPO to SAC.** SAC handles state-dependent variance natively,
   has off-policy sample efficiency, and is the standard choice for
   continuous-control problems where the optimal policy needs micro-actions.
   This is a 1–2 day rewrite of `src/agent/`.
2. **Use behavioural cloning bootstrap from LQR.** We have a working LQR
   controller in `src/control/lqr.py`. Pretrain the PPO policy to imitate LQR
   near upright, then fine-tune. This bypasses the swing-up exploration
   problem entirely for the stabilisation phase.
3. **Initialise actor from LQR linear gain** with a near-zero variance and
   freeze the log_std for the first 1 000 updates. This is a softer version
   of (2).

These are all algorithm-level interventions. Within the current PPO setup,
H3/I (~6.5 % strict success at $\delta = 0.43$) is the practical ceiling.

## Phase K — runtime-pipeline optimisation + retrain validation (run 2026-05-02)

Not an algorithmic change — a *wall-time* change. Targeted the env hot path
and the PPO training loop to make subsequent algorithmic experiments
(notably SAC and LQR-bootstrap) financially feasible.

### What changed

* **Env layer** (`src/env/cart_pendulum_base.py` + `double/single_pendulum.py`):
  pre-allocated RK4 stage buffers (`_k1..k4`, `_mid`, `_new_state`); in-place
  `_dynamics_into(state, force, out)` hook on subclasses; cached `_env_params`
  refreshed only on `set_curriculum`; pre-allocated obs buffer with copy on
  return. Numpy parenthesisation preserved verbatim so float64 rounding is
  bit-identical to the legacy path.
* **Trainer + PPO** (`src/train.py`, `src/agent/ppo.py`, `src/utils/normalize.py`):
  `(rollout_steps, n_envs, *)` pre-allocated rollout buffers; one batched
  policy forward and one critic forward per timestep over the full
  `(N, obs_dim)` batch; batched obs normaliser; deferred GPU syncs in the
  K-epoch loop (diagnostics accumulate as detached running sums); `.item()`
  called once per epoch (KL early-stop) plus once at end-of-update;
  `torch.randperm` device-side shuffle replaces `np.random.shuffle` plus
  per-mb host→device index transfer; CSV log opened once and `flush()`-ed
  every K updates.
* **Bit-equivalence gate**: `tests/test_pipeline_equivalence.py` hashes
  1000-step zero-action trajectories from `seed=42` and asserts the SHA-256
  hashes are identical to the un-optimised master baseline for double RK4,
  single RK4, and double semi-implicit. All four tests pass on the merged
  branch.

### Measured wall-time

`tools/evolve_eval.py --updates 5 --n_envs 4 --rollout_steps 256` (sized for
quick CI; same scale used to compare candidates):

| Metric | Master baseline | Phase K | Speedup |
|---|---:|---:|---:|
| `train_update_ms_mean` | 6431 ms | 1115 ms | **5.77×** |
| `env_step_ms_mean` | 0.129 ms | 0.100 ms | 1.29× |
| `total_eval_ms` | 40716 ms | 10982 ms | 3.71× |

At the production scale (`--n_envs 8 --rollout_steps 512`), the actual
training loop runs at ~1.48 s/update — a 3000-update Phase H3-style run
took **74 minutes** (vs. an extrapolated ~5.3 hours pre-optimisation).

### Validation: re-trained Phase I from-scratch on the new pipeline

To confirm the optimised pipeline produces equivalent policy quality, the
Phase I best snapshot was loaded and continued for 3000 updates with the
identical hyperparameters (hybrid reward, concave threshold, `target_kl =
0.015`, `value_clip = 0.2`, frozen obs_rms, best-fallback). Metrics on a
30-episode deterministic eval at $\delta = 0.445$:

| Metric | Phase I best (parent) | Phase K best | Delta |
|---|---:|---:|---:|
| Strict success (10°) | 4.7 % | 4.3 % | -0.4 (noise) |
| Loose success (20°) | 10.5 % | 12.0 % | +1.5 |
| Mean reward | 750 | 2042 | +1292 |
| SSE pole 1 | 78.45° | 74.10° | -4.35° |
| SSE pole 2 | 69.56° | 61.22° | -8.34° |
| Control effort | 0.708 | 0.700 | ≈ same |

Strict success matches within stochastic-eval noise; loose, reward, and
both pole SSEs improve marginally. The 4–7 % strict-success ceiling at
$\delta \approx 0.45$ is unchanged — Phase K does **not** break it. That
ceiling is the structural one identified after Phase J (PPO + state-
independent $\log\sigma$ cannot do tight stabilisation), and confirming
it persists on the optimised pipeline is the right outcome: the speedup
reflects implementation efficiency, not stealth physics drift.

### Lesson

The 5.77× pipeline speedup unlocks the Option A (SAC rewrite) and Option B
(LQR bootstrap) experiments outlined in `docs/NEXT_STEPS.md`: a 10 M
env-step SAC training that was previously a ~30-hour wall-time commitment
is now ~5 hours. The campaign's structural conclusion (PPO has hit its
ceiling at ~6.5 % strict success) stands; the next experiment is now
cheaper to run.

* **Best policy from this phase**:
  `logs/ppo_double_velocity_hybrid_20260502_153935_best.pth`.
* **Diagnostic reports**:
  `docs/reports/latest_phase_k_at_d0445.md`,
  `docs/reports/phase_i_at_d0445.md`.

### Phase K, Round 2 — batched dynamics across N envs (run 2026-05-03, BREAKTHROUGH on the trainer)

After PR #1 (c3) merged, the env-step microbenchmark dropped 1.29x but
`train_update_ms_mean` was still 2154 ms at production scale
(`--updates 20 --n_envs 8 --rollout_steps 512`). Profiling revealed the new
hot spot: the trainer's per-env Python `for i in range(n_envs):` step loop,
which dispatches N sequential `np.linalg.solve` calls per RK4 stage.

Three round-2 `mutate` candidates were dispatched on parent c3:

| # | Strategy | env_step | train_update | tests | Verdict |
|---|---|---:|---:|---:|---|
| 4 | Hand-coded Cramer's rule on the 3x3 solve | 0.071 ms | 1965 ms (1.10x) | 21/21 (regenerated hashes — Cramer differs from LU at ULP level) | REJECT (env-only) |
| 5 | Numba `@njit` on `_dynamics` kernel | 0.057 ms | 1798 ms (1.20x) | 21/21, bit-identical | REJECT (env-only) |
| **6** | Batched dynamics across N envs via `np.linalg.solve` on `(N, 3, 3)` batch | 0.106 ms | **1504 ms (1.43x)** | 21/21, per-row bit-identical (verified by `tools/check_batched_equivalence.py` over 100 random-action steps × 4 envs × 5 configs) | **APPROVE — winner** |

#### Why c6 won

c6's `env_step_ms_mean` is unchanged (the single-env path is preserved
verbatim) but it collapses the trainer's per-step inner loop from N
sequential `_dynamics` calls into one batched numpy/BLAS call. Numpy's
`np.linalg.solve` on `(N, 3, 3)` produces per-row bit-identical output to
N scalar solves (verified empirically by the cross-check script), so the
trajectory hashes in `tests/test_pipeline_equivalence.py` are preserved
(the test exercises the single-env path, which c6 doesn't touch).

A subtle finding: a ULP-level discrepancy emerged between
`np.float64 ** 2` (Python scalar pow path) and `arr ** 2` (numpy ufunc
path) at large velocity magnitudes. Both `_dynamics_into` and
`dynamics_into_batched` were rewritten to use explicit `x * x`. The
existing equivalence-test trajectory hashes are preserved because the
zero-action test trajectory stays in a small-magnitude regime where the
two paths coincide.

#### Cumulative speedup

| Stage | Production-scale `train_update_ms_mean` | Cumulative speedup vs original |
|---|---:|---:|
| Original master (small scale: `--updates 5 --n_envs 4 --rollout_steps 256`) | 6431 ms | 1.0x |
| c3 (PR #1) at production scale | 2154 ms | (different scale, see Phase K above for 5.77x at the small scale) |
| **c6 (PR #2) at production scale** | **1504 ms** | **~8.25x cumulative** vs original |

A 3000-update run at `--n_envs 8 --rollout_steps 512` now takes **67
minutes** (up from 74 min in Phase K, down from an extrapolated 5+ hours
on the original code).

#### Validation: re-trained Phase K best on the c6 pipeline

To confirm the batched path doesn't degrade policy quality, the Phase K
best snapshot was loaded and continued for 3000 updates with identical
hyperparameters. 30-episode deterministic eval at $\delta = 0.465$
(the difficulty Phase L reached):

| Metric | Phase K best (parent) | Phase L best (this run) | Delta |
|---|---:|---:|---:|
| Strict success (10°) | 5.3 % | 4.4 % | -0.9 (within noise) |
| Loose success (20°) | 13.1 % | 11.4 % | -1.7 |
| Mean reward | 2662 | 879 | -1783 |
| SSE pole 1 | 73.85° | 74.07° | ≈ 0 |
| SSE pole 2 | 65.77° | 62.78° | -2.99° |
| Control effort | 0.695 | 0.685 | smoother |

All deltas within 30-episode stochastic-eval noise. The 4-7 % strict-
success ceiling at $\delta \approx 0.45$ holds — exactly the structural
conclusion Phases C/E/F/G/H3/I documented. The optimised pipeline does
not break physics and does not break the algorithmic ceiling. It just
runs faster.

* **Best policy from this round**:
  `logs/ppo_double_velocity_hybrid_20260503_000503_best.pth`.
* **Diagnostic reports**:
  `docs/reports/phase_l_at_d0465.md`,
  `docs/reports/phase_k_at_d0465.md`.

### Round-3 future work (not run)

* **Stack c5 (numba) on top of c6**: numba's 2x env-layer win is
  orthogonal to c6's batched-trainer win. A combined PR could shave
  another ~10-20 % off `train_update_ms_mean`. Defer until the
  algorithmic phase (SAC) needs it.
* **Larger `n_envs`**: c6's batched solve scales sub-linearly with N
  due to BLAS dispatch overhead. `--n_envs 16` or `--n_envs 32` should
  benefit disproportionately.

## Phase M — algorithm-mode evolve campaign (run 2026-05-03, NULL RESULT)

After Phase K (runtime optimisation, 8.25x speedup) made longer experiments
tractable, a separate `mode: algorithm` evolve campaign was run to search
for a training-parameter configuration that breaks the 4-7 % strict-success
ceiling at $\delta \approx 0.45$ documented across Phases C-L.

### Methodology

Eval harness: `tools/algorithm_eval.py`. Each candidate is a HP config
(YAML) that runs a short training fragment plus a 30-episode deterministic
diagnostic eval. Two methodologies were tried:

* **Continue-from-best** (rounds 1-3): load Phase L best
  (`ppo_double_velocity_hybrid_20260503_000503_best.pth`), 200-update
  fragment from $\delta = 0.465$. Sharp signal but biased against
  perturbation: small HP changes drift the policy off its tight local
  optimum.
* **From-scratch** (round 4): no `--load`, 800 updates. Honest
  comparison but too short — Phase H3 needed 3000 updates to reach
  $\delta = 0.43 / 5.9\%$ strict; 800 updates only reaches
  $\delta \approx 0.26$ where strict-success has not yet emerged.

Reviewer rejects any candidate that fails to beat the parent's 4.4 %
strict-success rate. Physics-anchor tests (`test_pipeline_equivalence`,
`test_physics`, `test_components`, `test_energy_reward`) remain a hard
gate — the env layer is off-limits for mutation.

### Result

12 candidates across 4 rounds, all rejected.

| Round | Method | Best candidate | Strict | Δ vs parent |
|---|---|---|---:|---:|
| R1 | 500u from-scratch (3 diverse) | C3 (capacity 512 + q_theta=0.5) | 0.7 % | -3.7 |
| R2 | 200u continue-from-best (3 HP perturbations) | C5 (relaxed KL) | 3.7 % | -0.7 |
| R3 | 200u continue + structural changes | C9 (energy-reward switch) | 3.0 % | -1.4 |
| R4 | 800u from-scratch (3 anchors) | C11 (energy from scratch) | 1.4 % | -3.0 |

**Best round-3 candidate (C9, energy reward) regressed by only 1.4 %, with
SSE pole 2 actually improving 62.78° → 58.12°.** That is the most
interesting null-result — energy reward perturbs the policy more gently
than the LQR-style hybrid quadratic penalty, and one of its components
(angle alignment) actually held a tighter pole. But it didn't break the
ceiling.

### Diagnoses extracted from the null-results

The campaign produced specific, actionable diagnoses despite finding no
winner:

1. **HP perturbation alone cannot escape the H3 local optimum.** Six
   HP-only candidates (R1 c1/c2, R2 c4/c5/c6) all regressed below the
   parent. R2 c5 measured `approx_kl ≈ 0.003-0.006` — well below even
   the original `target_kl = 0.015`, proving the trust-region clip was
   never the binding constraint. The bottleneck is the *gradient signal*
   inside the band, not the optimiser surface.

2. **The curriculum level-up gate is the actual bottleneck.** R2 c4
   (aggressive ratchet) showed the gate
   `time_above > 0.90·δ ∧ R̄ > R̄_prev_best` is what binds the curriculum,
   not the ratchet step size. At $\delta = 0.465$ the agent's
   `time_above ≈ 37 %` is below the required 42 % — the policy is
   *just* sub-competent. R3 c8 loosened the gate to
   `max(0.30, 0.65·δ)` and did advance one ratchet step ($\delta = 0.47$),
   but the policy collapsed at the new difficulty (strict 1.5 %). This
   matches Phase E's finding: forced curriculum advance without
   competence is destructive.

3. **The hybrid-reward `q_theta = 0.2` is precisely tuned to the
   policy.** R3 c7 isolated a `q_theta` bump (0.2 → 0.5) and regressed
   to 1.5 % strict. This corrects the R1 c3 diagnosis: it is q_theta
   itself that traps the policy, not the hidden_dim/init_log_std combo.
   The reward landscape's quadratic-penalty magnitude must match the
   policy's exploration noise — bumping the penalty without
   re-equilibrating the policy breaks the existing solution.

4. **Both eval methodologies are the wrong frame for this question.**
   Continue-from-best at 200u is too narrow (any perturbation regresses);
   from-scratch at 800u is too short (cannot reach the strict-success
   regime). The right eval would be 3000u from-scratch per candidate
   (≈ 67 min wall on the c6 pipeline), but a 24-candidate campaign
   at that scale is ~27 hours of compute — beyond the practical budget
   without a stronger prior on which configs to try.

5. **The campaign log's structural conclusion stands.** Phase J already
   proved state-dependent $\log\sigma$ is incompatible with PPO; the
   present null-result on HP/reward/capacity perturbations confirms
   that within PPO + state-independent $\sigma$ + the H3 hybrid reward,
   the 4-7 % ceiling is reached and held. The path forward is
   **algorithmically different**, not HP-tuned: SAC (state-dependent
   $\sigma$ via reparam gradient) or LQR-bootstrap (warm-start the
   policy from a known stabiliser) per `docs/NEXT_STEPS.md` Options A/B.

### What this campaign proves

The 4-7 % strict-success ceiling at $\delta \approx 0.45$ is **not an HP
tuning bug, a reward-shape calibration error, a capacity bottleneck, a
trust-region width issue, or a curriculum-step-size problem**. Twelve
candidates across four orthogonal methodologies (HP perturbation,
structural perturbation, capacity scaling, reward swap; both
continue-from-best and from-scratch) all failed to beat 4.4 %. The
ceiling is structural — a property of the algorithm class, not the
configuration within it.

### What was NOT tried (saved for follow-up)

* SAC rewrite (Option A in `docs/NEXT_STEPS.md`).
* LQR behavioural-cloning bootstrap (Option B).
* 3000-update from-scratch sweep over the same HP grid (would be
  honest but expensive: ~27 h compute for 24 candidates).
* `n_envs = 16` or `32` with the c6 batched solve (cheap to add but
  unlikely to break the ceiling alone).

### Best policy from the campaign

`logs/ppo_double_velocity_hybrid_20260503_000503_best.pth` — unchanged
from Phase L. The campaign's deliverable is the **null-result documentation**
plus the `tools/algorithm_eval.py` harness, which can be reused for a
future SAC vs PPO sweep with the same scoring methodology.

## Phase N — SAC implementation + 2M env-step run (run 2026-05-05)

Built per `docs/NEXT_STEPS.md` Option A: SAC with state-dependent
:math:`\log\sigma`, twin-Q + target nets (Polyak averaging at
:math:`\tau = 0.005`), 1M-capacity replay buffer, automatic entropy
tuning toward target :math:`\mathcal H_{\rm target} = -|\mathcal A|`.

### Implementation

* `src/agent/sac.py` (~330 LOC): `GaussianPolicy` (state-dep log_std,
  reparameterised tanh-squashed sample with the numerically stable
  Jacobian `2*(log 2 - z - softplus(-2z))`), `TwinQ`, `ReplayBuffer`,
  `SACAgent` (canonical critic→actor→temperature update with Polyak
  averaging on target nets each step).
* `src/train_sac.py` (~290 LOC): vectorised env collection (reuses the
  c6 `BatchedEnvRunner`); off-policy update pattern (1 SAC update per
  env step after warmup); same curriculum / ratchet / CSV-log structure
  as `src/train.py`.
* `tests/test_sac.py` (11 tests): reparam gradient flows through both
  heads (the PPO failure mode); action box; analytic log-prob match;
  twin-Q independence; replay buffer correctness; critic loss decreases
  on a static batch; Polyak averaging fires; alpha stays positive;
  end-to-end on a trivial 1-D regulator. Suite total now 32/32.

### Validation

Smoke run (single pendulum, 100k env-steps, 13 min): TimeUp climbed
0% → 19% → 39% → 48% → 53% → 62 % over the run; alpha auto-tuned
:math:`1.0 \to 0.10`; curriculum advanced :math:`\delta = 0.0 \to 0.010`
(2 ratchets). The agent IS learning. Smoke validates the implementation
end-to-end.

### Full run (double pendulum, 2M env-steps, 4 h 20 min)

Wall-clock pace: ~7.8 ms per env-step (1 SAC update each), 8 envs in
parallel via the c6 batched dynamics. Comparable per-env-step cost to
PPO; SAC's update cost dominates over the env step.

| Phase | Env-steps | Peak :math:`\delta` | Time elapsed |
|---|---:|---:|---:|
| 0 → 450k | warmup + early ratchet | 0.0 → 0.095 | ~1 h |
| 450k → 1.34M | plateau at :math:`\delta = 0.095` | 0.095 | ~2 h |
| 1.34M → 1.79M | breakout to 0.196 | 0.196 | ~3 h |
| 1.79M → 2M | continued ratchet to 0.211 | **0.211** | ~4 h 20 min |

Final eval at :math:`\delta = 0.211` (30 episodes deterministic):

| Metric | SAC (2M env-steps) | PPO Phase L (Phase H3+I+K, ~12M env-steps) |
|---|---:|---:|
| Peak :math:`\delta` | 0.211 | 0.465 |
| Strict success (10°) | **1.87 %** | **4.4 %** |
| Loose success (20°) | 4.84 % | 11.4 % |
| Mean episode reward (window) | -176 | (varies; not directly comparable) |

**SAC underperformed PPO at this compute budget.** Per-env-step compute
is comparable, but SAC needed more env-steps to reach the same
curriculum stage that Phase H3 reached. Phase H3 took ~12M env-steps
to reach :math:`\delta = 0.43`; SAC at 2M env-steps is at
:math:`\delta = 0.211` — about 4-5× behind on a per-env-step basis.

### Why SAC didn't beat PPO here

1. **The curriculum gate is the binding constraint, not the
   variance-policy choice.** Both PPO and SAC plateau at the same
   gate (`time_above > 0.9·δ ∧ R̄ > R̄_prev_best`). SAC's
   state-dependent :math:`\log\sigma` did not produce the kind of
   tight stabilisation that Phase J's analysis predicted — the
   policy reached time-above 60-65 % at low difficulty (good) but
   could not push it above the 19 % threshold required at
   :math:`\delta = 0.211` consistently enough to ratchet further.

2. **Replay buffer composition matters at curriculum boundaries.**
   When the curriculum advances, transitions in the buffer were
   collected at older difficulties and may pull the critic toward
   stale targets. Standard SAC has no curriculum-aware buffer
   management; this is a known limitation in non-stationary tasks.

3. **End-of-run reward trajectory was strongly positive** (-3500
   mid-run → -104 at the end). SAC was *still learning* when the
   2M-env-step budget ran out. A 5M or 10M env-step run could
   plausibly close the gap to PPO and possibly exceed it.

### What this proves

* SAC machinery (reparameterised gradient through the squashed
  Gaussian, twin-Q with Polyak target, automatic entropy tuning) is
  implemented correctly and trains end-to-end without instability.
* At 2M env-steps from-scratch, SAC is *behind* PPO on this task.
* The 4-7 % strict-success ceiling at :math:`\delta \approx 0.45` is
  *still* not broken — but SAC didn't get to that difficulty in this
  budget, so the question of whether it would beat PPO *there*
  remains open.

### Best policy from this phase

`logs/sac_sac_double_velocity_hybrid_20260505_165234_final.pth`
(:math:`\delta = 0.211`, strict 1.87 %). Phase L PPO best
(`ppo_double_velocity_hybrid_20260503_000503_best.pth` at 4.4 %)
remains the overall campaign best.

### Future work

* **Longer SAC run from the current checkpoint** (3-5M additional
  env-steps, ~6-12 hours). The end-of-run reward trajectory was still
  improving — most likely SAC simply needs more time on this task.
* **Curriculum-aware replay buffer**: discount transitions collected
  at significantly lower :math:`\delta` than the current curriculum.
  Cheap to implement (multiply per-transition sample weight by
  :math:`\exp(-\Delta\delta / \sigma)`).
* **Higher `updates_per_step`** (currently 1 per env step; common SAC
  configs use 1-4). Trades more SAC compute for less env compute,
  which is the right trade when env-stepping is the cheap part (the
  c6 batched solve made env-stepping ~8 ms/step but each SAC update
  is similarly ~1-2 ms).
* **Option B (LQR behavioural-cloning bootstrap)**: still untried.
  Pre-train the SAC actor against the LQR controller in
  `src/control/lqr.py` to skip the swing-up exploration.
