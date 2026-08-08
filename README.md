# Artifact: an explicit kwarg vs a provider's registry default

`capture_wbc_walk_flag.py` drives the public `build_policy_kwargs` helper with
`walk=False` (documented as "only the main policy"), builds the WBC policy from
whatever kwargs come back, and runs an 8 s MuJoCo rollout of the Unitree G1
headless (`MUJOCO_GL=egl`) with a 0.6 m/s forward velocity command. It records
which ONNX session actually ran, the pelvis trajectory, and a frame every 0.5 s.

Run once per tree (`upstream/main` in a worktree, then the branch), then
`compose_figure.py` builds the figure and asserts every number it renders.
