# Artifacts: a failed Zenoh session close reported as a clean one

- `session_close_report.png` -- the published figure.
- `measure.py` -- drives `release_session()` / `_atexit_cleanup()` with a session whose
  `close()` raises each of ZError / OSError / TypeError / AttributeError, and records the
  log output, whether the call raised, and whether the session reference was dropped.
  Run once per tree; each run prints the tree it imported from.
- `mutate.py` -- the mutation table: 6 plausible regressions x 2 arms (this PR's tests vs
  upstream's own copy of the same module). Anchors are scoped to the enclosing function by
  AST line range and the source is restored byte-identically.
- `census.py` -- the coverage census that selected this module.
