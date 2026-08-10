# Reachy transport degradation branches -- measurement scripts

`capture.py` re-derives every cell of the figure: the two full-suite coverage
snapshots for `reachy_transport.py`, the measured behaviour of each of the three
tolerance branches, and the 5x2 mutation matrix (each mutation applied inside the
target function's own AST line range, then restored byte-identically).

`compose.py` renders the figure and asserts every number it draws against
`art_facts.json` before saving, plus a per-side white-border check.

Run with `PYTHONPATH=<repo> python3 capture.py` from the repo root.
