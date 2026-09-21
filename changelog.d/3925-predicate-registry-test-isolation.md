### Fixed: a predicate a test registers no longer outlives it

`PREDICATE_REGISTRY` is process-global, so a name `register_predicate` added
during one test was still there for every later test in the process, and the
grader that reads the registry as the set of shipped predicates failed on a
name only a test knew. The test session now restores the registry between
cells, replacing seven per-site `try`/`finally` cleanups.
