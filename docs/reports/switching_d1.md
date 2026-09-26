# Equilibrium switching evaluation (δ = 1)

Model-based switching between the four equilibria (`src/control/switching.py`): one multiple-shooting trajectory per ordered pair, TVLQR tracking, LQR hold at each equilibrium. Env: `DoublePendulumCartEnv`, full difficulty, wind σ = 1 N, `ForceControl`; 10 seeds per transition. Success = cart within |x| ≤ 3.5 m throughout and both poles within 0.17 rad of the target for the whole final second (5 s after the nominal transition ends).

| From → to | Direction (Δθ₁, Δθ₂) | T [s] | Nominal peak F [N] | Success | Reached after [s] | Peak F [N] | Final error P1 / P2 |
|---|---|---:|---:|---:|---:|---:|---:|
| DD → UU | (+π, +π) | 4.0 | 26.4 | 100 % | 4.01 | 27 | 0.04° / 0.01° |
| DD → DU | (0, +π) | 4.0 | 11.3 | 100 % | 4.01 | 12 | 0.04° / 0.01° |
| DD → UD | (+π, 0) | 4.0 | 23.2 | 100 % | 4.01 | 24 | 0.03° / 0.02° |
| UU → DD | (+π, +π) | 4.0 | 26.3 | 100 % | 4.39 | 27 | 0.04° / 0.03° |
| UU → DU | (−π, 0) | 4.0 | 26.9 | 100 % | 4.39 | 27 | 0.04° / 0.01° |
| UU → UD | (0, +π) | 4.0 | 31.5 | 100 % | 4.39 | 33 | 0.03° / 0.02° |
| DU → DD | (0, −π) | 4.0 | 11.3 | 100 % | 4.41 | 17 | 0.04° / 0.03° |
| DU → UU | (+π, 0) | 4.0 | 26.5 | 100 % | 4.41 | 27 | 0.04° / 0.01° |
| DU → UD | (−π, −π) | 4.0 | 19.0 | 100 % | 4.41 | 20 | 0.03° / 0.02° |
| UD → DD | (−π, 0) | 4.0 | 61.0 | 100 % | 4.01 | 61 | 0.04° / 0.03° |
| UD → UU | (0, −π) | 4.0 | 31.5 | 100 % | 4.01 | 33 | 0.04° / 0.01° |
| UD → DU | (−π, −π) | 4.0 | 14.7 | 100 % | 4.01 | 16 | 0.04° / 0.01° |

**Tour** (all 12 transitions in one episode, 3 s dwell at each stop): 5 / 5 completed.

"Reached after" is measured from the request to the first step at which the controller is holding the target with both poles in the strict band. Direction ±π is the way each pole turns over; 0 = keeps its orientation.

Command: `python tools/eval_switching.py --rebuild --workers 1 --seeds 10 --report docs/reports/switching_d1.md`
