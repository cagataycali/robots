# Artifact: report a teleop auto-accept that could not answer the prompt

`teleop_auto_accept.png` — measured on Thor (NVIDIA Jetson AGX Thor).

- `capture.py` — drives `lerobot_teleoperate(action="start", auto_accept_calibration=True)`
  against a recording stub child process in both trees and dumps the measured
  outcomes to JSON (start status, session-store pid, stdin bytes delivered, log records).
- `compose.py` — builds the figure from those two JSON dumps. Every rendered number is
  asserted against the dumps before the PNG is saved.
- `measure.py` — the 4-row outcome probe (write succeeds / write raises, on each tree).
- `mutate.py` — the 7-row mutation table, two arms (the new module vs the 58
  pre-existing `tests/tools/test_lerobot_teleoperate.py` cases). Each anchor is
  AST-scoped to its enclosing function and the source is restored byte-identically.
