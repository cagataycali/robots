### Changed: a provider's class is imported by the policy factory, not by the registry

`import_policy_class` read `policies.json` and then fell back to scanning
`strands_robots.policies.<name>` for a `Policy` subclass - an `issubclass`
against a class two layers above the declarative registry, deferred behind a
late import so the runtime graph never showed it. It lives in
`strands_robots.policies.factory` now, where `Policy` is already a module-level
name, and is exported from `strands_robots.policies`; the redundant second
registry lookup inside it is gone, since `get_policy_provider` already keys on
the canonical name. `strands_robots.registry` no longer exports the name and no
re-export is left behind, so a caller importing it from there must import it
from `strands_robots.policies`. Deferred upward import inversions 11 -> 10 in
`scripts/check_import_layers.py`, whose roster line is deleted rather than
suppressed, with a new pin in `tests/test_import_layers_are_a_dag.py` asserting
no module under `strands_robots.registry` reads a policy in any import kind.
Towards #3818.
