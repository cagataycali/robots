### Tests: the registry suite has one home, and a copied module is graded

`tests/test_registry.py` and `tests/test_registry_integrity.py` were copies of
`tests/registry/test_public_api.py` and `tests/registry/test_integrity.py` that
had drifted apart, and being outside the `tests/registry` package they ran
without its user-registry isolation fixture. Both are removed, the seven cells
only they held moved into `tests/registry/test_integrity.py`, and a new
whole-tree grader refuses a module that shares more than half of its test
bodies with any single other module.
