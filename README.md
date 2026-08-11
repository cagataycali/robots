# repr must not hide the constructor refusal that produced the object

`capture.py` runs on one tree and dumps every measurement to JSON: the verbatim
pytest render of a refused `RosBridgedRobot`, the half-built-instance survey over
every class in the package that defines `__repr__`, the fully-built reprs, an AST
digest of each touched module with its `__repr__` and utils import removed, and a
headless MuJoCo render.

`compose.py` reads the two dumps, asserts every number it draws (including that
the two arms measured different trees) and writes the figure.

    PYTHONPATH=<tree> MUJOCO_GL=egl python3 capture.py out.json   # once per tree
    MUJOCO_GL=egl python3 compose.py
