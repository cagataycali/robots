# Cosmos 3: explicit de-normalization stats must declare their domain

Measured on Thor (MUJOCO_GL=egl, Franka panda + mink IK). `capture.py` runs in a
tree and dumps JSON; `compose.py` builds the figure and asserts every rendered
number against the two dumps; `sweep.py` chose the camera.

One `nvidia/Cosmos3-Edge` `umi` action chunk (16 steps x 10 columns), decoded three ways:

| call | main | this PR |
|---|---|---|
| umi's own quantiles + `stats_domain='umi'` | `TypeError` (no such keyword) | decoded, 0.2055 m |
| `bridge_orig_lerobot` quantiles, domain declared | decoded, 0.2614 m (+27.2%) | refused, names both domains |
| `bridge_orig_lerobot` quantiles, no domain | decoded, 0.2614 m (+27.2%) | refused, names `stats_domain` |

Honored render identical across trees (max |delta| = 1/255); the defect panel
differs on 15.79% of pixels.
