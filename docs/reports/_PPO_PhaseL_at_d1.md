# Training Diagnostic Report
**Timestamp**: 2026-05-08 04:00:09
**Model**: `ppo_double_velocity_hybrid_20260503_000503_best.pth`
**Evaluation difficulty (delta)**: 1.000 (g = 9.810, mu_cart = 0.000, threshold = 10.0 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **2.4%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 6.6% | % time both poles within 20° of upright |
| **Avg Reward** | -24 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 77.60° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 63.32° | Avg error over the trailing 20% |
| **Control Effort** | 0.682 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
