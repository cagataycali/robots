### Tests: a test that writes no `STRANDS_*` variable no longer pays for the environment restore

The autouse fixture that puts `STRANDS_*` variables back after every test was
meant to cost one copy and one compare when nothing changed. pytest rewrites
`PYTEST_CURRENT_TEST` between a test's setup and its teardown, so the compare
never matched and every test decoded the whole environment twice. That one
variable is now compared at its teardown value; 6,000 empty tests take 0.6 s
less with a 100-variable environment.
