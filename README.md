# max_onframe_failures domain - measurement artifact

Generated on a MuJoCo headless host (`MUJOCO_GL=egl`), so100, 2.0 s at 50 Hz.

* `capture.py` - run in a checkout of each tree; drives `PolicyRunner.run` with a
  hook that renders one frame per step, and with a hook that raises every call.
  Prints the tree it resolved so a before/after pair cannot be two runs of one tree.
* `compose.py` - reads the two `facts_*.json` dumps and asserts every number it
  renders before saving the figure.
* `probe_domain_table.py` - the full value/verdict table for the parameter.
