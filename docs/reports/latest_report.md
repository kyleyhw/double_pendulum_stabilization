# Training Diagnostic Report
**Timestamp**: 2026-05-02 16:54:58
**Model**: `ppo_double_velocity_hybrid_20260502_153935_best.pth`
**Evaluation difficulty (delta)**: 1.000 (g = 9.810, mu_cart = 0.000, threshold = 10.0 deg)

## Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Strict Success Rate** | **4.0%** | % time both poles within 10° of upright |
| **Loose Success Rate** | 9.1% | % time both poles within 20° of upright |
| **Avg Reward** | -36 | Mean episode reward |
| **Avg Length** | 4000 | Mean episode steps |
| **Steady-State Error (P1)** | 75.08° | Avg error over the trailing 20% |
| **Steady-State Error (P2)** | 64.94° | Avg error over the trailing 20% |
| **Control Effort** | 0.689 | Mean |action| in normalised units |

## How to read this
* Strict success rate >80% indicates solid stabilisation.
* Steady-state error <5° is excellent; 5–10° acceptable.
* Control effort: lower is smoother (less bang-bang).
