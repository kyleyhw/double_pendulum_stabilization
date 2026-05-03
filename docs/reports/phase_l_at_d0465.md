# Training Diagnostic Report
**Timestamp**: 2026-05-03 01:13:16
**Model**: `ppo_double_velocity_hybrid_20260503_000503_best.pth`
**Evaluation difficulty (delta)**: 0.465 (g = 5.632, mu_cart = 0.267, threshold = 52.8 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **4.4%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 11.4% | % time both poles within 20° of upright |
| **Avg Reward** | 879 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 74.07° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 62.78° | Avg error over the trailing 20% |
| **Control Effort** | 0.685 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
