# Phase S policy evaluation (δ = 1)

Model `logs/phaseS_seed1_full.pth` (curriculum reached: {'r': 0.25, 'reach': 4.0, 'reach_max': 4.0}); F_max = 100 N, action repeat 4 (20 ms). 100 episodes × 20 s per column on `DoublePendulumCartEnv` (wind σ = 1 N), deterministic policy.

| Metric | Basin r = 0.25 rad | From hanging |
|---|---:|---:|
| Survived | 96.0 % | 100.0 % |
| Stayed within soft cart bound | 95.0 % | 100.0 % |
| Terminal-1 s strict (mean) | 95.0 % | 100.0 % |
| Episodes with terminal-1 s strict = 100 % | 95 / 100 | 100 / 100 |
| Max sustained strict (mean) | 19.28 s | 17.60 s |
| Final-second error P1 / P2 | 7.36° / 8.02° | 0.02° / 0.00° |
| Peak cart force | 100 N | 100 N |

Command: `python tools/eval_balance.py --model logs/phaseS_seed1_full.pth --episodes 100 --radius 0.25`
