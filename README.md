# Validated mirror: an import referenced only inside a string

Backs strands-labs/robots#2114 round 2.

`unused_import_mirror.py` reimplements the "unused import" rule as
"an imported name never referenced by an `ast.Name` or `ast.Attribute`
anywhere in the module", excluding the `__future__` compiler directive.

## Validation

Run against the file the scanner flagged, it reproduces the published alert
exactly - line and column range included:

    $ python3 unused_import_mirror.py tests/policies/test_state_key_name_list_contract.py
    tests/policies/test_state_key_name_list_contract.py:81:1-37	Callable
    --- 1 finding(s) over 1 file(s) ---

The published alert is `tests/policies/test_state_key_name_list_contract.py:81`,
columns 1-37. Because it reproduces a verdict the scanner had already published,
its verdict on the candidate fix is trustworthy: after unquoting the alias element
the same command reports `--- 0 finding(s) over 1 file(s) ---`.

## Scope measurement quoted in the PR

    $ python3 unused_import_mirror.py tests
    tests/policies/test_state_key_name_list_contract.py:81:1-37	Callable
    tests/simulation/isaac/test_send_action_action_value_domain.py:41:1-72	fake_isaacsim_types
    tests/simulation/test_policy_runner.py:31:1-22	strands_robots
    --- 3 finding(s) over 905 file(s) ---

The second is a pytest fixture imported for injection (referenced by parameter
name, not by expression); the third is already open on `main`. Neither is the
string-annotation shape and neither is new in this diff, so the fix is one line.

Note the mirror deliberately does not model `__all__` re-exports, so it
over-reports inside packages that re-export; it is scoped to the `tests/` tree,
where that pattern does not occur.
