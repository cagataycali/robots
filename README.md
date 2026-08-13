# Artifact: a hardware rollout dispatch failure reports its own cause

`capture.py` measures one tree (run it once per tree, `PYTHONPATH` pinned to that
tree); `compose.py` builds `artifact.png` from the two dumps and asserts every
number it draws.

    PYTHONPATH=<tree> MUJOCO_GL=egl python3 capture.py <outdir> <label>
    MUJOCO_GL=egl python3 compose.py <outdir>

Measured on Thor (arms unplugged; the driver is an in-memory fake, the render is
headless MuJoCo).
