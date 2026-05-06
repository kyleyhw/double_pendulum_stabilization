# Training Diagnostic Report
**Timestamp**: 2026-05-06 01:18:56
**Model**: `ppo_double_velocity_hybrid_20260503_000503_best.pth`
**Evaluation difficulty (delta)**: 0.271 (g = 4.117, mu_cart = 0.364, threshold = 68.3 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **5.2%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 13.9% | % time both poles within 20° of upright |
| **Avg Reward** | 2750 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 69.79° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 65.58° | Avg error over the trailing 20% |
| **Control Effort** | 0.682 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
