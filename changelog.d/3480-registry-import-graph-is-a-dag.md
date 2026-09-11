### Changed: the registry package's modules no longer import in a cycle

`loader` and `user_registry` imported each other for the user-overlay reads, and
`user_registry` reached `robots`, which imports `loader`. Both cycles are cut by
a new leaf, `strands_robots.registry._overlay`, that owns the `user_robots.json`
path, read and parse and imports no registry sibling. The imports those cycles
had forced into function bodies - including an `except ImportError: pass` that
could silently drop the overlay merge - are now module-scope. The public surface
is unchanged: `user_registry_source` and `parse_user_robots` still import from
`strands_robots.registry`. A new grader,
`tests/registry/test_import_graph_is_acyclic.py`, refuses a new intra-package
cycle, counting function-local imports the way CodeQL's `py/cyclic-import` does.
