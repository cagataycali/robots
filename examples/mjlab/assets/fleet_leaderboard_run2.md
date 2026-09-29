# MuJoCo-Warp Reach Policy Leaderboard
**Run date:** 2026-09-29 · **Steps:** 200 PPO iterations · **Envs:** 1 024 · **Eval:** 20 CPU episodes each

| Rank | Arm | DoF | Train time (min) | Train `at_goal` | Eval success | Median final error (mm) | Status |
|------|---------|-----|-----------------|-----------------|--------------|------------------------|--------|
| 1 | koch | 6 | 7.46 | 0.6979 | 19/20 | 6.8 | ✅ OK |
| 2 | arx_l5 | 6 | 5.87 | 0.5646 | 17/20 | 15.0 | ✅ OK |
| 3 | so101 | 5 | 6.71 | 0.6123 | 14/20 | 12.6 | ✅ OK |

> **DoF** counts actuated joints only (gripper excluded).  
> **Train `at_goal`** = `Metrics/reach/at_goal` reported at the final PPO iteration.  
> **Median final error** = median Euclidean end-effector distance to target at episode tick 150, across 20 seeded episodes.  
> Success threshold: 30 mm for so101 & koch; 61.2 mm for arx_l5 (arm-specific, as returned by `evaluate_policy`).
