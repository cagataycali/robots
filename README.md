# Measured artifact: two second-line guards on the mesh wire-authorisation path

`capture.py` runs inside the branch checkout, re-derives every number the figure
shows (the 5x2 mutation table plus the coverage accounting read from two
full-suite `--cov` runs) and writes `measured.json`. `compose.py` renders the
figure and asserts each rendered value against that dump before saving, so a
stale panel cannot ship.

Reproduce:

    PYTHONPATH=<checkout> python3 _art/capture.py <cov-main.json> <cov-pr.json> measured.json
    python3 _art/compose.py measured.json mesh-second-line-guards.png
