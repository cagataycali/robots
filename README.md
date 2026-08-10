# LIBERO `max_steps` domain -- measurement scripts

`capture.py` measures, on one tree:

* what each of the four public `max_steps` surfaces does with an unusable
  horizon (the three adapter surfaces raise; `load_libero_suite` skips each task);
* a mutation table -- three regressions applied to the guard in
  `LiberoAdapter.__init__`, each run against the new tests and against the
  pre-existing `tests/benchmarks/libero` suite, with every anchor AST-scoped to
  `__init__`'s own line range and the source restored byte-identically;
* the coverage delta for `adapter.py:541` read from two `--cov` json reports.

`compose.py` renders `max_steps_domain.png` and asserts every rendered number
against the capture dump before saving.

Run as `MUJOCO_GL=egl PYTHONPATH=<tree> python3 capture.py` from the repository
root; both scripts print the tree they resolved `strands_robots` from.
