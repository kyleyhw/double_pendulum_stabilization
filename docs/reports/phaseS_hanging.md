# Phase S policy evaluation (δ = 1)

Model `logs/phaseS_seed0_hanging.pth` (curriculum reached: {'r': 0.25, 'tau_min': 0.0}); F_max = 100 N, action repeat 4 (20 ms). 100 episodes × 20 s per column on `DoublePendulumCartEnv` (wind σ = 1 N), deterministic policy.

| Metric | Basin r = 0.25 rad | From hanging |
|---|---:|---:|
| Survived | 100.0 % | 99.0 % |
| Stayed within soft cart bound | 100.0 % | 98.0 % |
| Terminal-1 s strict (mean) | 100.0 % | 99.0 % |
| Episodes with terminal-1 s strict = 100 % | 100 / 100 | 99 / 100 |
| Max sustained strict (mean) | 19.74 s | 17.27 s |
| Final-second error P1 / P2 | 1.91° / 1.79° | 3.64° / 3.56° |
| Peak cart force | 100 N | 100 N |

## Stress tests

Swing-up from hanging, 10 seeds per cell; success as in `docs/robustness_report.md` (inside the soft bound, strict for the final second).

| Cart impulse 8 s in [N·s] | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Success | 100 % | 100 % | 100 % | 0 % |

| Wind σ [N] | 1 | 5 | 10 | 20 | 40 |
|---|---:|---:|---:|---:|---:|
| Success | 100 % | 100 % | 100 % | 90 % | 90 % |

| Parameter | ×0.8 | ×0.9 | ×1.1 | ×1.2 |
|---|---:|---:|---:|---:|
| M | 100 % | 100 % | 100 % | 100 % |
| m1 | 100 % | 100 % | 100 % | 80 % |
| m2 | 80 % | 80 % | 100 % | 100 % |
| l1 | 20 % | 60 % | 100 % | 100 % |
| l2 | 10 % | 0 % | 20 % | 10 % |

Command: `python tools/eval_balance.py --model logs/phaseS_seed0_hanging.pth --episodes 100 --radius 0.25 --stress`
