### Fixed: CodeQL scans the package, not the test suites

`.github/codeql/codeql-config.yml` gains `paths-ignore` for `tests/` and
`tests_integ/`. 69 of the 197 alerts open on 2026-09-28 were note-severity
findings in test code (unused fixture imports, a swallowed teardown error, a
duplicate import) that no wheel user can reach and that the pull-request
graders and ruff already read. Each was dismissed as won't-fix with the reason
written in the config file. The two query filters and the pin test on them are
unchanged.
