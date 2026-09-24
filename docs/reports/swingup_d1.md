# Designed swing-up evaluation (δ = 1)

Reset `down`, horizon 20 s, wind σ = 1 N, seeds 0–99. Strict band: both poles within 0.17 rad of upright.

Trajectory: T = 4.00 s, goal angles = (+3.1416, +3.1416), peak nominal force 26.4 N, peak nominal |x| 2.50 m, max defect 4.8e-11.

| Metric | Value |
|---|---:|
| Episodes | 100 |
| Survived | 100.0 % |
| Stayed within soft cart bound | 100.0 % |
| Terminal-1 s strict (mean) | 100.0 % |
| Episodes with terminal-1 s strict = 100 % | 100 / 100 |
| Max sustained strict (mean) | 15.51 s of 20 s |
| First strict entry (mean) | 4.49 s |
| Steady-state P1 / P2 | 0.038° / 0.016° |
| Peak cart force (max) | 27 N |

Command:

```bash
python tools/eval_swingup.py --reoptimize --save_traj src/control/data/swingup_d1.npz --episodes 100 --report docs/reports/swingup_d1.md
```
