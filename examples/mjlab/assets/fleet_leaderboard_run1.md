# Reach Policy Leaderboard — MuJoCo-Warp Fleet (2026-09-29)

> Sorted by evaluation success (descending). Training: rsl_rl, 200 steps, batch 1024, seed 0.
> Success threshold: 30 mm. Evaluation: 20 seeded episodes, CPU MuJoCo backend.

| Rank | Arm | DoF | Train wall time (min) | `Metrics/reach/at_goal` (train) | Eval success | Median final error (mm) | Status |
|------|--------|-----|----------------------:|--------------------------------:|:------------:|------------------------:|--------|
| 1 | koch | 6 | 7.97 | 0.7188 | 19 / 20 | 6.4 | ✅ OK |
| 2 | so101 | 5 | 6.51 | 0.5906 | 14 / 20 | 15.5 | ✅ OK |
| 3 | arx_l5 | 7 | 23.11 | 0.4222 | — / 20 | — | ❌ `ValueError: model has 7 outputs but 8 joint_names / 7 action_scale entries` |
