# Equilibrium switching evaluation (δ = 1)

Goal-gated mixture of four RL experts `logs/phase6_center_DD_final.pth,logs/phase6_center_UU_final.pth,logs/phase6_center_DU_final.pth,logs/phase6_center_UD_final.pth` (`src/train_balance.py --goals expert:<NAME>`, one per target equilibrium, selected by the requested goal; a switch starts once the current equilibrium is held with |x|, |ẋ| < 0.3). The T and nominal-force columns describe the switching library the policy was trained along (start states only), not what the policy does. Env: `DoublePendulumCartEnv`, full difficulty, wind σ = 1 N, `ForceControl`; 10 seeds per transition. Success = cart within |x| ≤ 3.5 m throughout and both poles within 0.17 rad of the target for the whole final second (5 s after the nominal transition ends).

| From → to | Direction (Δθ₁, Δθ₂) | T [s] | Nominal peak F [N] | Success | Reached after [s] | Peak F [N] | Final error P1 / P2 |
|---|---|---:|---:|---:|---:|---:|---:|
| DD → UU | (+π, +π) | 4.0 | 26.4 | 100 % | 2.36 | 94 | 0.39° / 0.37° |
| DD → DU | (0, +π) | 4.0 | 11.3 | 100 % | 2.24 | 99 | 0.78° / 0.59° |
| DD → UD | (+π, 0) | 4.0 | 23.2 | 100 % | 2.37 | 99 | 0.17° / 0.17° |
| UU → DD | (+π, +π) | 4.0 | 26.3 | 100 % | 2.13 | 99 | 0.36° / 0.36° |
| UU → DU | (−π, 0) | 4.0 | 26.9 | 100 % | 0.66 | 99 | 0.03° / 0.04° |
| UU → UD | (0, +π) | 4.0 | 31.5 | 100 % | 2.57 | 96 | 0.05° / 0.05° |
| DU → DD | (0, −π) | 4.0 | 11.3 | 100 % | 1.41 | 100 | 0.12° / 0.13° |
| DU → UU | (+π, 0) | 4.0 | 26.5 | 100 % | 1.94 | 99 | 0.27° / 0.27° |
| DU → UD | (−π, −π) | 4.0 | 19.0 | 100 % | 1.38 | 99 | 0.11° / 0.11° |
| UD → DD | (−π, 0) | 4.0 | 61.0 | 100 % | 0.56 | 100 | 0.04° / 0.03° |
| UD → UU | (0, −π) | 4.0 | 31.5 | 100 % | 2.66 | 75 | 0.15° / 0.13° |
| UD → DU | (−π, −π) | 4.0 | 14.7 | 100 % | 1.21 | 100 | 0.11° / 0.12° |

**Tour** (all 12 transitions in one episode, 3 s dwell at each stop): 5 / 5 completed.

"Reached after" is measured from the request to the first step at which the controller is holding the target with both poles in the strict band. Direction ±π is the way each pole turns over; 0 = keeps its orientation.

Command: `python tools/eval_switching.py --policy logs/phase6_center_DD_final.pth,logs/phase6_center_UU_final.pth,logs/phase6_center_DU_final.pth,logs/phase6_center_UD_final.pth --seeds 10 --report docs/reports/switching_rl_d1.md`
