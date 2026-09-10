### Changed: `validate_command` states its contract by refusal shape

The docstring of `strands_robots.mesh.security.validate_command` names the four shapes a refusal takes and each refusal code once, instead of restating every rule the body enforces. The test that graded that rule-by-rule census against the function's AST is removed; its fifteen behavioural tests (bool refused, non-finite refused, envelope re-read per call, teleop identifier charset, `override_code` bounds) move unchanged to `tests/mesh/test_validate_command_value_and_action_rules.py`. No code path changes.
