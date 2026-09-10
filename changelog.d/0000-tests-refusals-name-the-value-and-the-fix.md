### Added: a test that every customer-path refusal quotes the bad value and names the fix

`tests/test_refusals_name_the_value_and_the_fix.py` drives `Robot("so100")`
through 13 wrong calls a first-day user actually makes (negative `tol`, a 2-d
target, a typo'd gripper state, `n_steps=-1`, an unknown body, ...) and asserts
that each refusal quotes the offending value, contains a fix (a range, the valid
literals, a did-you-mean, or the listing action), and leaves the simulation
state unchanged. `test_error_paths.py` locks "error, not exception"; this locks
the next layer the model reads. Tests only.
