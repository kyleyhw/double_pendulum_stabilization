# Next Steps — Self-Contained Handoff

**Updated: 2026-07-12.** This supersedes the 2026-05-01 version of this file
(which recommended the SAC rewrite — since executed, Phases K–Q). It is the
resume point for the cart-double-pendulum project and is written to be
readable in a fresh session with no prior context. The chronological campaign
log is `docs/EXPERIMENTS.md` (currently ends mid-Phase-P; see
[Repository state](#repository-and-environment-state)).

## Executive summary

The question "why doesn't the stabilization work?" was answered definitively
on 2026-07-12:

1. **The plant and the balance layer are correct and solved.** The equations
   of motion were re-derived from the Lagrangian and match the implementation
   term-by-term. A plain LQR at the up-up equilibrium, run through the actual
   simulator at full difficulty ($\delta = 1$), holds both poles upright
   indefinitely (100 % strict over the whole 20 s horizon from small
   perturbations, steady-state error < 0.1°).
2. **The swing-up-to-capture step is the sole blocker, and every approach
   tried to date — PPO, SAC, LQR-bootstrapped SAC, and the classical
   energy-shaping hybrid — fails there for the same structural reason:**
   each drives the system onto the correct *energy level* and implicitly
   assumes it will then visit the upright configuration. That assumption is
   a theorem for one pole and false for two: on the 2-link chain the
   constant-energy set is chaotic, and the joint event "both poles
   simultaneously within the LQR capture band with small velocities"
   essentially never occurs spontaneously.

The recommended fix is a designed swing-up (trajectory optimization +
tracking, Option A below) or a reverse curriculum that starts learning inside
the verified LQR basin (Option B). Tuning the existing rewards or gains
further attacks a structural problem with parametric tools and is not
recommended.

## Where we are

* **Honest metrics** (introduced in Phase O): a *balance* policy must survive
  the horizon, score terminal-1 s strict near 100 % (fraction of the final
  second with both poles within 10° of upright), and sustain multi-second
  strict dwells. Whole-episode strict — the metric behind earlier "champion"
  numbers — is misleading and inflated by short crashy episodes.
* **No policy to date balances.** By the honest metrics, at their own
  difficulties: PPO survives but never gets upright (max sustained strict
  0.18 s); SAC transits upright (~1.0 s peak dwell) but always crashes the
  cart; energy-reward SAC (Phase P) and LQR-buffer-bootstrapped SAC
  (Phase Q) survive with poles tumbling near horizontal (steady-state errors
  94–99°, terminal-1 s strict ≤ 0.2 %).

| Checkpoint | What it is | Honest status |
|---|---|---|
| `logs/ppo_double_velocity_hybrid_20260503_000503_best.pth` | best PPO on headline task ($\delta=1$) | survives, never balances |
| `logs/sac_sac_double_velocity_hybrid_20260506_231612_final.pth` | "26.21 % strict" SAC champion ($\delta=0.295$) | metric artifact — transits, crashes |
| `logs/sac_sac_double_velocity_energy_20260514_170753_best.pth` | Phase Q best (energy reward + LQR bootstrap) | tumbling-horizontal optimum |

## The 2026-07-12 diagnostic (unlogged; write up as Phase R)

### Blocker found first

`scipy` (declared in `pyproject.toml`) was missing from `venv`, so
`tools/eval_lqr.py`, `tools/eval_hybrid.py`, and everything importing
`src/control/lqr.py` crashed on import. These tools were written in the last
hour of the 2026-05-14 session and had therefore **never been run** — the
decisive experiment below is what that session was about to do. scipy 1.18.0
(and pytest) were installed on 2026-07-12; all 32 tests pass (13.9 s).

### Experiments run

All at full difficulty $\delta = 1.0$ ($g = 9.81$, frictionless, wind
$\sigma_w = 1$ N). Strict band = both poles within 0.17 rad (≈ 10°) of
upright; the LQR handoff band in the hybrid is 15° enter / 25° exit.

```bash
# 1. LQR from tiny perturbations (basic stabilizability test), 20 episodes
python tools/eval_lqr.py --difficulty 1.0 --episodes 20 \
    --init_radius_theta 0.03 --init_radius_thetadot 0.05 \
    --init_radius_x 0.1 --init_radius_xdot 0.05
# 2. LQR from default larger perturbations (basin probe), 30 episodes
python tools/eval_lqr.py --difficulty 1.0 --episodes 30
# 3. Full hybrid (energy swing-up -> LQR) from hanging, 15 episodes
python tools/eval_hybrid.py --difficulty 1.0 --episodes 15
```

| Test | Survived | Terminal-1 s strict | Max sustained strict | Steady-state P1/P2 |
|---|---:|---:|---:|---:|
| LQR, ±1.7° / ±0.05 rad s⁻¹ | 100 % | **100 %** | 20.00 s (= horizon) | 0.08° / 0.04° |
| LQR, ±14.3° / ±0.5 rad s⁻¹ / ±1 m | 76.7 % | **79.2 %** | 14.95 s | 26.6° / 26.3° |
| Hybrid from hanging | 100 % | **0.0 %** | 0.00 s | 102.0° / 98.1° |

How to read this: row 1 says the linearization, gain, and simulator are all
correct — LQR parks both poles upright to within a tenth of a degree and
never lets go. Row 2 maps the basin of attraction: even from simultaneous
14° errors on both poles plus velocity and cart offsets, LQR captures ~79 %
of episodes — the capture region a swing-up must hit is *not* small. Row 3 is
the failure: in 15 episodes **the swing-up never once brought both poles
within the 15° handoff band** (`Episodes that engaged LQR: 0 / 15`), so the
LQR — which row 1 proves would hold — was never given the state it needs.

### Energy-budget trace of one hybrid episode

Method: one episode, seed 123, $\delta = 1.0$, `swingup_k = 80`, sampled
every 1 s; total energy from `hybrid_controller._total_energy_double_with_M`,
pole-only energy = same expression minus the cart term
$\tfrac12 M\dot x^2$. Target $E_0^{UU} = 2[(m_1{+}m_2)gl_1 + m_2gl_2] =
29.43$ J (for $m_1{=}m_2{=}0.5$ kg, $l_1{=}l_2{=}1$ m, $g{=}9.81$ m s⁻²).

| $t$ [s] | $E_{\rm tot}$ [J] | $E_{\rm pole}$ [J] | err₁ [°] | err₂ [°] |
|---:|---:|---:|---:|---:|
| 0 | 0.01 | 0.01 | 177 | 178 |
| 1 | 29.41 | 27.54 | 128 | **9.9** |
| 2 | 29.43 | 29.42 | **9.6** | 176 |
| 3 | 29.43 | 28.45 | 48 | **3.5** |
| 7 | 24.67 | 16.24 | 138 | 148 |
| 11 | 8.26 | 8.01 | 177 | 100 |
| 13 | 29.43 | 29.43 | 160 | 92 |
| 19 | 29.39 | 20.67 | 127 | 134 |

How to read this: the energy law *works as an energy regulator* — within
1 s the system is on the target level $E \approx E_0$ with ~94 % of the
energy in the poles (so the "cart KE steals the budget" hypothesis is
refuted). But look at the error columns: each pole individually passes
within 4–10° of upright (bold), **always while the other is far away**
(t = 2 s is a textbook up-down configuration). Over the full 20 s the best
*simultaneous* approach was $\max(e_1, e_2) = 41.6°$ — never near the 15°
handoff band. Secondary defect: the dips at t = 7 and t = 11 (E down to
8.3 J) are the cart-recentering guard at $|x| > x_{\rm soft}$ bleeding
energy that then must be re-pumped; the swing-up law has no cart-position
term, so this cycle repeats indefinitely.

## Root cause

For a cart force $F$, total mechanical energy obeys $\dot E = F \dot x$
(frictionless case), so energy can be regulated to any target — that part is
easy and works. The failure is in what energy regulation *buys you*:

* **One pole** (1 DOF once the energy is fixed): the level set
  $\{E = E_0\}$ of the pole subsystem **is** the homoclinic orbit through
  the inverted equilibrium. Coasting on it delivers the pole to upright with
  zero velocity. "Pump energy, then catch" is therefore a complete strategy
  — this is the classical result of Åström & Furuta
  [[1]](#ref-astrom-2000).
* **Two poles** (the pole subsystem is 4-D): $\{E = E_0\}$ is a 3-D set on
  which the dynamics are chaotic. The up-up saddle's stable manifold has
  positive codimension inside the level set, so the trajectories that
  approach up-up form a measure-zero subset; a generic trajectory wanders
  through up-down/down-up configurations forever — exactly what the trace
  shows. Energy at the right level is **necessary but not sufficient**, and
  an energy controller has no authority over *where on the level set* the
  state goes.

Every approach tried so far instantiates this same flaw:

1. **Classical hybrid** (2026-05-14): pure energy shaping → 0/15 captures,
   demonstrated above.
2. **`EnergyShapingReward` RL** (Phases B, P): the reward's energy term
   $w_E \exp(-\Delta E^2 / 2\sigma_E^2)$ pays out *for manifold membership*.
   A policy that pumps to $E_0$ and lets the poles tumble collects it in
   full, forever — the Phase P signature (100 % survival, steady-state
   errors ≈ 95°, i.e. tumbling) is this reward's actual optimum. The capture
   bonus exists but is gated behind an event exploration never samples.
3. **`HybridLQRSwingUpReward` RL** (Phases H3–L): broke the intra-band
   gradient problem but has the documented horizontal local optimum at loose
   curriculum thresholds (Phase O), and its "champions" transit rather than
   balance.
4. **LQR replay-buffer bootstrap** (Phase Q): pre-filling SAC's buffer with
   LQR balance transitions gave the critic the missing high-value states and
   *still* produced 0.2 % terminal strict — evidence that value-side
   demonstration is insufficient when the policy has no path into the basin.
   The blocker is reaching the capture set, not valuing it.
5. **Compounding, RL-specific**: the training envs use `VelocityControl`
   ($F = K_p(a\,v_{\max} - \dot x)$ with $K_p = 10^4$, $v_{\max} = 10$), so
   force resolution is $\partial F / \partial a = K_p v_{\max} = 10^5$ N per
   unit action. Balance-scale corrections of ~10 N correspond to action
   increments of $10^{-4}$ — three orders of magnitude below the exploration
   noise ($\sigma \approx 0.3$–$0.5$). Policies could never *experience*
   sustained balance during training. The verified LQR results above used
   `ForceControl`.

## Recommended next steps, ranked

### Option A — designed swing-up: trajectory optimization + tracking (recommended if the goal is "solve the system")

This is the standard, literature-validated solution for this exact plant:
Graichen, Treuer & Zeitz swung up a real cart double pendulum with a
feedforward trajectory + feedback tracking [[2]](#ref-graichen-2007); the
method is textbook material [[3]](#ref-tedrake).

**Scope** (~2 sessions):
1. `src/control/swingup_trajopt.py` — direct collocation over the verified
   dynamics: decision variables $\{s_k, u_k\}_{k=0}^{N}$, trapezoidal or
   Hermite–Simpson defect constraints, boundary conditions DD-rest →
   UU-rest, bounds $|x| \le 3.5$, $|F| \le F_{\max}$, objective
   $\int u^2\,dt$. Horizon $T \in [2, 5]$ s (the linearized UU periods are
   1.0 s and 2.0 s; Graichen et al. used comparable durations — treat $T$ as
   a search parameter). Solver: `scipy.optimize.minimize(SLSQP)` may
   suffice at $N \approx 100$; CasADi + IPOPT is the robust convenience
   alternative (new dependency — decide before adding).
2. Time-varying LQR tracking: backward differential Riccati along the
   nominal trajectory, reusing the existing finite-difference linearization
   from `src/control/lqr.py`.
3. Terminal handoff to the existing (verified) `LQRController`; the
   hysteresis logic in `src/control/hybrid_controller.py` is reusable as-is.
4. Evaluate with `tools/eval_hybrid.py` metrics. Success = terminal-1 s
   strict ≳ 90 % at $\delta = 1.0$ from hanging.

### Option B — RL with a reverse curriculum from the LQR basin (recommended if the goal is "RL solves it")

Convert the measure-zero exploration problem into a sequence of easy local
problems [[4]](#ref-florensa-2017): initialize episodes *inside* the
verified basin (±14° already gives LQR 79 % capture; RL should exceed this
since it is not restricted to linear feedback), train SAC with
**`ForceControl`** to hold, then ratchet the initial-state radius outward
toward hanging.

**Scope** (~1 session to first milestone): an `--init_mode basin --init_radius r`
reset option; curriculum on $r$ instead of (or alongside) physics difficulty;
SAC otherwise as in Phase K. First milestone: ≥ 95 % terminal-1 s strict at
$r = 0.25$ rad, $\delta = 1.0$ — should be quick, and immediately produces
the project's first genuinely balancing learned policy. The endgame (full
swing from hanging) may still want Option A's trajectory as a demonstration
source; the two options compose.

### Option C — capture-shaped reward on the existing pipeline (cheapest, lowest ceiling)

Keep everything, but pay dwell time inside the LQR-capturable set
explicitly (e.g. bonus per consecutive step with both poles < 15° and
$|\dot\theta_i|$ small) instead of energy-manifold membership. Attacks the
reward side of the structural problem but leaves exploration to luck; try
only as a rider on Option B.

## Repository and environment state

* **Environment**: `venv` was missing declared dependencies. scipy 1.18.0
  and pytest were installed 2026-07-12. The dev group (`ruff`, `ty`,
  `pre-commit`, `detect-secrets`) is still not installed in `venv`; sync
  before relying on hooks.
* **Tests**: 32/32 pass (2026-07-12, 13.9 s).
* **Uncommitted work from 2026-05-14** (sitting in the working tree):
  terminal-bound-penalty env/train changes, `src/control/equilibria.py`,
  `src/control/hybrid_controller.py`, `src/agent/lqr_bootstrap.py`,
  `tools/eval_lqr.py`, `tools/eval_hybrid.py`, `tools/rollout_trace.py`,
  ~30 reports under `docs/reports/`, Phase O/P additions to
  `docs/EXPERIMENTS.md`. Needs a commit decision.
* **Log gap**: `docs/EXPERIMENTS.md` ends mid-Phase-P. Phase P final and
  Phase Q exist only as `docs/reports/v2_phaseP_final_d025.md` /
  `v2_phaseQ_best_d006.md`. Today's diagnostic is unlogged — write it up as
  Phase R when appending.
* **Curriculum physics** (needed to interpret any $\delta$):
  $g(\delta) = 2 + 7.81\,\delta$, $\mu_{\rm cart} = 0.5(1-\delta)$,
  $\mu_{\rm pole} = 0.1(1-\delta)$, wind $\sigma_w = \delta \cdot 1$ N.
  $\delta = 1$ is full gravity, frictionless.

## What NOT to redo

* Don't re-verify the plant or the LQR — done 2026-07-12, numbers above.
  The dynamics match a from-scratch Euler–Lagrange derivation; the
  equilibria eigenstructure (`docs/reports/equilibria_*.md`) is correct.
* Don't tune the energy swing-up (`swingup_k`, bands, guard gains) — the
  failure is structural (chaotic level set), not parametric.
* Don't build another reward that pays for energy-manifold membership or
  band proximity without capture — Phases B/P/Q are three refutations.
* Don't use `VelocityControl` for balance-phase work; the action-resolution
  argument above. `ForceControl` is verified.
* Don't retry state-dependent log-std under PPO (Phase J; algorithmic, not
  tuning) and don't headline whole-episode strict (Phase O).

## Concrete starting prompts

Option A:

> Read `docs/NEXT_STEPS.md`. Implement Option A: direct-collocation
> swing-up trajectory for the cart double pendulum at $\delta = 1.0$ over
> the dynamics in `src/env/double_pendulum.py`, then TVLQR tracking and
> handoff to the existing `LQRController` (verified working — see the
> 2026-07-12 diagnostic). Evaluate with `tools/eval_hybrid.py` metrics;
> target terminal-1 s strict ≥ 90 % from hanging.

Option B:

> Read `docs/NEXT_STEPS.md`. Implement Option B: SAC with `ForceControl`
> and a reverse curriculum on the initial state, starting inside the LQR
> basin (±0.25 rad, ±0.5 rad/s — LQR captures 79 % there; see the
> 2026-07-12 diagnostic) and expanding toward hanging. First milestone:
> ≥ 95 % terminal-1 s strict at radius 0.25 rad, $\delta = 1.0$.

## References

<span id="ref-astrom-2000">[1]</span> Åström, K. J., & Furuta, K. (2000).
*Swinging up a pendulum by energy control.* Automatica, 36(2), 287–295.
[Link](https://doi.org/10.1016/S0005-1098(99)00140-5)

<span id="ref-graichen-2007">[2]</span> Graichen, K., Treuer, M., & Zeitz, M.
(2007). *Swing-up of the double pendulum on a cart by feedforward and
feedback control with experimental validation.* Automatica, 43(1), 63–71.
[Link](https://doi.org/10.1016/j.automatica.2006.07.023)

<span id="ref-tedrake">[3]</span> Tedrake, R. *Underactuated Robotics:
Algorithms for Walking, Running, Swimming, Flying, and Manipulation.*
Course notes, MIT. [Link](https://underactuated.mit.edu/)

<span id="ref-florensa-2017">[4]</span> Florensa, C., Held, D., Wulfmeier,
M., Zhang, M., & Abbeel, P. (2017). *Reverse curriculum generation for
reinforcement learning.* CoRL 2017. [Link](https://arxiv.org/abs/1707.05300)
