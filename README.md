# Isaac `add_camera(parent_body=...)` — reporting the unsupported camera mount

`capture.py` runs the Isaac refusal (no Isaac Sim needed: it precedes the lock) and
the same mount on the backend the refusal names, on a real headless MuJoCo so101.
`compose.py` builds the figure; every drawn number is asserted against `facts.json`
before the PNG is written, including that the mounted wrist view really changes
when the body it rides moves (38.91% of pixels over 90 applied actions).

Reproduce: `MUJOCO_GL=egl PYTHONPATH=<robots checkout> python3 capture.py && python3 compose.py`
