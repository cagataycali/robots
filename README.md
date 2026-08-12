# fromto-fixed geom_size components

`capture.py` runs once per tree (upstream/main in a detached worktree, then the
branch) and dumps `facts-*.json` plus the raw frames. `compose.py` reads both
dumps, asserts every number it renders, and writes the figure.

    MUJOCO_GL=egl PYTHONPATH=<tree> python3 capture.py <outdir>
    python3 compose.py <outdir> <main-tag> <pr-tag>

Nothing is hand-typed: the panel captions, the table and the pixel percentages
are all derived from the two dumps, and `compose.py` fails if any claim drifts.
