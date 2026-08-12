# move_to / rotate_wrist not-reached envelope

Measurement artifacts for the motion-primitive result-envelope tests.

- `capture.py` - drives a real MuJoCo headless world (the motion-primitive
  suite's inline arm), renders the three states and dumps `facts.json`. Every
  claim in the figure is asserted in this script before it is written.
- `compose.py` - builds `move_to_not_reached_envelope.png` from `facts.json`.
- `frame_sweep.py` - the camera sweep used to pick the framing (all three
  panel pairs must differ on more than 10% of pixels).
- `mutate.py` - the 7-regression mutation table, run against the new cases and
  against the pre-existing suite.

Run with `MUJOCO_GL=egl PYTHONPATH=. python3 capture.py` from a repo checkout.
