### Fixed: the `seed` contract of `run_policy` and `eval_policy` states what reproducible means on GPU rendering

The docstrings promised the same trajectory on re-run and the rollouts page
said seeded evaluations replay the same episodes, but under `MUJOCO_GL=egl` a
static scene renders with 1 LSB differences between frames and a seeded ACT
rollout drifts from them (0.006 to 0.3 rad after 90 to 150 steps), while a
state-only policy re-runs bit-exact. All three texts now say so, naming the
renderer and telling the reader to compare outcomes rather than frames, and a
grader holds them to it. (#4192)
