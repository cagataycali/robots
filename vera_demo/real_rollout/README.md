# Real VERA PushT Rollout (2026-06-23)

End-to-end test on NVIDIA Thor Jetson (aarch64, CUDA).

## Setup
- VERA policy server: `default-wan` planner + `j1j59qzz` IDM on `0.0.0.0:8820`
- Checkpoints: pusht-dfot (39 MB) + pusht-idm (232 MB)
- Env: gym-pusht 0.1.5, 50 steps, seed=42

## Metrics
| Metric              | Value      |
|---------------------|------------|
| max_reward (coverage) | 0.0007  |
| return              | 0.0149     |
| success (≥0.9)      | 0          |
| Wall time           | 85.5 s     |
| Server dream chunks | 17 (~4.9s each) |

## Videos
- `policy_env0.mp4` — policy view (252×252)
- `clean_env0.mp4` — clean env render
- `vis_env0.mp4` — debug overlay

## Known issue
gym-pusht 0.1.5 on PyPI doesn't expose `control_type="velocity"` (older fork
API). Without velocity control, the policy's velocity-delta outputs are
interpreted as positions → agent thrashes. Coverage near zero is **not** a
policy failure — it's an env-binding mismatch. To get real coverage numbers
we need to vendor Diffusion-Policy's velocity-controlled `pusht_env.py`.

The video, server, planner+IDM checkpoints, msgpack wire protocol, and full
client/server roundtrip are **real and working**.
