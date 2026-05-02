# Training Diagnostic Report
**Timestamp**: 2026-05-02 16:56:57
**Model**: `ppo_double_velocity_hybrid_20260501_184759_best.pth`
**Evaluation difficulty (delta)**: 0.445 (g = 5.475, mu_cart = 0.277, threshold = 54.4 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **4.7%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 10.5% | % time both poles within 20° of upright |
| **Avg Reward** | 750 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 78.45° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 69.56° | Avg error over the trailing 20% |
| **Control Effort** | 0.708 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
