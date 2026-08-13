# bare output filename anchors to the sandbox

`capture.py` measures the contract on one tree (run with `PYTHONPATH=<tree>`);
`compose.py` builds the figure and asserts every rendered number against the two
JSON dumps. `mutate.py` is the mutation table (5 regressions x 2 arms).

Measured on Thor, headless (`MUJOCO_GL=egl`), against upstream/main 83cc5272.
