### Fixed: a name bound by a module-scope import is not imported again inside a function

Thirteen function-scope imports across six package modules repeated a
binding the same module already makes at module scope - `asyncio` in
`hardware_robot._run_control_loop`, `numpy as np` five times in the Isaac
backend, `torch`, `os`, `dataclasses`, `importlib.util` and two `from`
imports elsewhere. Each was a `sys.modules` lookup that shadowed a global
with an identical local, and each read as a deliberately deferred import
with a reason to look for. CodeQL reports the shape as
`py/repeated-import`; the `hardware_robot` instance had been open on `main`
since May and opened a merge-gating review thread on every branch that
added lines above it. Pinned by
`tests/test_module_scope_imports_are_not_repeated_in_functions.py`, a
whole-tree grader that leaves the deferred (`TYPE_CHECKING`) and
optional-dependency (`try`/`except ImportError`) patterns alone.
