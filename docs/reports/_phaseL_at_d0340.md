# Training Diagnostic Report
**Timestamp**: 2026-05-08 03:58:03
**Model**: `ppo_double_velocity_hybrid_20260503_000503_best.pth`
**Evaluation difficulty (delta)**: 0.340 (g = 4.655, mu_cart = 0.330, threshold = 62.8 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **3.6%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 9.6% | % time both poles within 20° of upright |
| **Avg Reward** | 806 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 74.04° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 62.66° | Avg error over the trailing 20% |
| **Control Effort** | 0.688 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
