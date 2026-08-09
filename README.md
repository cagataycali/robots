# Artifact: a hardware task's policy_port is judged before the arm is connected

`capture.py` drives `Robot.start_task` against an in-memory arm with a recording
connect path and dumps every measured fact to JSON. Run it once in a checkout of
`main` and once on the branch; `compose.py` reads both dumps, asserts they came
from different trees, re-derives every number it renders, and refuses to save a
figure whose panels contradict the measurement.

    python3 capture.py facts_main.json     # in a main checkout
    python3 capture.py facts_branch.json   # on the branch
    python3 compose.py

No serial/USB hardware is touched.
