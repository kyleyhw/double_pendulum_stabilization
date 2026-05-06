# Training Diagnostic Report
**Timestamp**: 2026-05-06 19:48:29
**Model**: `ppo_double_velocity_hybrid_20260503_000503_best.pth`
**Evaluation difficulty (delta)**: 0.295 (g = 4.304, mu_cart = 0.353, threshold = 66.4 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **4.8%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 12.2% | % time both poles within 20° of upright |
| **Avg Reward** | 1334 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 67.41° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 56.93° | Avg error over the trailing 20% |
| **Control Effort** | 0.684 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
