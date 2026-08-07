# Artifact: SimEnv action_scale domain

`artifact_action_scale.png` - three real MuJoCo headless rollouts (MUJOCO_GL=egl) of a
two-joint arm, 60 env steps of a constant `[0.9, -0.7]` command, plus the measured
before/after verdict table for `action_scale`.

`scripts/capture.py` runs the rollouts and dumps facts.json + one .npy render per case;
run it once per tree (a `git worktree` at the base, and the branch). `scripts/compose.py`
builds the figure and asserts every number it prints against the two dumps.
