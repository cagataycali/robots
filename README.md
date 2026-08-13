# artifacts: isaac-primitive-policy-guard

Measurements for the Isaac motion-primitive / policy contention fix.

| file | what it is |
| --- | --- |
| `isaac-primitive-policy-guard.png` | the figure referenced from the PR |
| `measure.py` | drives the three primitives with `policy_running` set (already-running) and set mid-run, recording status, report text and PD target-set count per cell |
| `capture.py` | records the two conflicting command streams against the suite's articulation fake and replays the poses onto a MuJoCo arm carrying the same joint names (`MUJOCO_GL=egl`) |
| `sweep_cameras.py` | the camera sweep that chose the render viewpoint by measuring differing-pixel fraction and saturation |
| `compose.py` | builds the figure; re-derives every rendered number from the JSON dumps and asserts them |
| `mutate.py` | the mutation table: 6 plausible regressions x 2 arms |
| `facts-*.json`, `matrix-*.json` | the raw dumps, one pair per tree |

Reproduce (from a checkout of the branch, with a sibling worktree at the merge base):

```
PYTHONPATH=$PWD MUJOCO_GL=egl python3 capture.py     # in each tree
PYTHONPATH=$PWD python3 measure.py                   # in each tree
PYTHONPATH=$PWD python3 compose.py
PYTHONPATH=$PWD python3 mutate.py
```
