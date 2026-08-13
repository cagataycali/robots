# GL-probe latch: measurement scripts

`capture.py` — run once per tree with `PYTHONPATH` pinned to that tree:

    PYTHONPATH=<tree> MUJOCO_GL=egl python3 capture.py main|branch

It records how many `mujoco.Renderer` objects one `gl_available()` builds after a
`cache_clear()`, and renders one real offscreen frame. Prints its own resolved tree
so a measurement can never be attributed to the wrong checkout.

`compose.py` — builds `gl-probe-latch.png` from the two JSON dumps. Every number drawn is
read from them; it asserts the two arms measured different trees, that the render is
byte-comparable across trees, the derived row pitches and a clean 8 px border.

`mutations.py` — the 5-mutation x 2-arm table. Derives both arms from the AST (the
pre-existing arm from the base blob), scopes each anchor to its enclosing function and
prints `in_fn`/`in_file`, counts errors as well as failures, and restores byte-identically.
