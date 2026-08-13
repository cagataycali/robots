# Artifact: the MuJoCo finalizer completes its teardown at interpreter shutdown

`capture.py` runs in one tree and measures three things: the reported 3-sim
scenario in a child process (both with and without an explicit `cleanup()`),
which teardown steps the finalizer reaches at real interpreter shutdown, and one
headless render per robot. `compose.py` reads the two dumps, asserts every
number it draws, and refuses to save if the renders differ by more than 2/255 or
a border pixel is not white. `mutations.py` runs each regression against the new
tests and against the 19 cases already in the module.

Measured on Thor (NVIDIA Jetson AGX Thor, `MUJOCO_GL=egl`, mujoco 3.11.0).

    python3 capture.py <out> main    # in a worktree at the base
    python3 capture.py <out> pr      # in the branch
    python3 compose.py <out> mutations.json
