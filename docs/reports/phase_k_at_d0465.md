# Training Diagnostic Report
**Timestamp**: 2026-05-03 01:14:11
**Model**: `ppo_double_velocity_hybrid_20260502_153935_best.pth`
**Evaluation difficulty (delta)**: 0.465 (g = 5.632, mu_cart = 0.267, threshold = 52.8 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **5.3%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 13.1% | % time both poles within 20° of upright |
| **Avg Reward** | 2662 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 73.85° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 65.77° | Avg error over the trailing 20% |
| **Control Effort** | 0.695 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
