### Fixed: the registry package no longer imports in a cycle

`loader` and `user_registry` imported each other for three overlay reads, and
`user_registry` reached `robots`, which imports `loader`. Both cycles are cut by
a new leaf, `strands_robots.registry._overlay`, that owns the
`user_robots.json` path/read/parse and imports no registry sibling. Public
names are unchanged (`user_registry_source` and `parse_user_robots` still
resolve from `strands_robots.registry` and `.user_registry`). A grader,
`tests/registry/test_import_graph_is_acyclic.py`, refuses a new intra-package
cycle - function-local imports included, because that is what CodeQL's
`py/cyclic-import` counts - so the four registry alerts stop being re-attributed
to pull requests that only shift a line. Closes #3146 for `registry/`.
