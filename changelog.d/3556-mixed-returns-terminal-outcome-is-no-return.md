### Docs: a `py/mixed-returns` alert on a helper ending in a pytest outcome is a false positive, and the boundary is graded

`pytest.fail`, `pytest.skip`, `pytest.exit` and `pytest.xfail` are declared
`-> NoReturn`, so a test helper that returns a value on one path and ends in
one of them on the other has no implicit return; CodeQL does not model the
outcome and reported the shape three times, the third holding a merge.
`AGENTS.md` records the counterfactual (`mypy` reports a missing return on an
annotated helper once the outcome is replaced, and nothing for `-> Any` or an
unannotated one), and
`tests/test_mixed_return_helpers_end_in_a_pytest_outcome.py` refuses a helper
whose terminal call can return, whatever its annotation.
