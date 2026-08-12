# finite_vector_error: a vector whose length cannot be read

`capture.py` runs unchanged in a worktree at `upstream/main` and in the branch;
`compose.py` builds the figure and asserts every number it renders (including
that the two arms measured different trees). `blast.py` is the 16-row verdict
matrix over both guards, `mutate.py` the 5-mutation table.

    MUJOCO_GL=egl PYTHONPATH=<tree> python3 capture.py {main|pr}
    python3 compose.py
