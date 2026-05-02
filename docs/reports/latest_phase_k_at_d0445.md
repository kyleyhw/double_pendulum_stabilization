# Training Diagnostic Report
**Timestamp**: 2026-05-02 16:56:09
**Model**: `ppo_double_velocity_hybrid_20260502_153935_best.pth`
**Evaluation difficulty (delta)**: 0.445 (g = 5.475, mu_cart = 0.277, threshold = 54.4 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **4.3%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 12.0% | % time both poles within 20° of upright |
| **Avg Reward** | 2042 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 74.10° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 61.22° | Avg error over the trailing 20% |
| **Control Effort** | 0.700 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
