# teleop publish rate domain -- measurement artifacts

`capture.py` drives the three entry points named in
`tests/test_teleop_rate_and_duration_guards.py`'s module docstring with `hz=0`,
records what each does, and then runs the five mutations of
`Robot.start_teleop_publish` against two arms (the new test class, and the
pre-existing cases over the same four test files). Every number in the figure
comes from `measurements.json`; `compose.py` asserts each one before saving.

Run from a checkout of the PR branch:

    MUJOCO_GL=egl python3 _art/capture.py && python3 _art/compose.py
