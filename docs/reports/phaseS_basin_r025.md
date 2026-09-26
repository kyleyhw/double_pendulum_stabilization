# Phase S policy evaluation (δ = 1)

Model `logs/phaseS_seed0_basin.pth` (curriculum reached: {'r': 0.25, 'tau_min': 4.0}); F_max = 100 N, action repeat 4 (20 ms). 100 episodes × 20 s per column on `DoublePendulumCartEnv` (wind σ = 1 N), deterministic policy.

| Metric | Basin r = 0.25 rad | From hanging |
|---|---:|---:|
| Survived | 99.0 % | 91.0 % |
| Stayed within soft cart bound | 93.0 % | 2.0 % |
| Terminal-1 s strict (mean) | 99.0 % | 50.5 % |
| Episodes with terminal-1 s strict = 100 % | 99 / 100 | 48 / 100 |
| Max sustained strict (mean) | 19.08 s | 4.12 s |
| Final-second error P1 / P2 | 2.70° / 2.53° | 46.20° / 47.47° |
| Peak cart force | 100 N | 100 N |

Command: `python tools/eval_balance.py --model logs/phaseS_seed0_basin.pth --episodes 100 --radius 0.25`
