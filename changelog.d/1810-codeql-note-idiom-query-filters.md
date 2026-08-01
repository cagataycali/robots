### Fixed: a CodeQL note-severity idiom alert no longer blocks a merge

`.github/workflows/codeql.yml` stated that pull requests are not blocked on
CodeQL alerts. They were: `github-advanced-security` opens a review thread per
new alert and the `default` branch ruleset requires thread resolution, so every
new alert was a hard merge gate whatever its severity - two approved, green PRs
sat unmergeable on note-severity style findings alone. A new
`.github/codeql/codeql-config.yml` filters exactly two rules whose every
instance in this repository is an idiom the codebase is obliged to use:
`py/ineffectual-statement` (27 of 27 alerts were `...` as a `Protocol` /
`@abstractmethod` / `@overload` body) and `py/import-and-import-from` (63 of 64
were the pytest monkeypatch idiom, where the module alias is the patch target
and the `from` import names the subject). The real no-op-statement class is not
given up - ruff now selects `B015` and `B018`, moving it to a check that gates
merges where CodeQL is advisory. `py/empty-except` is deliberately untouched
and still requires a human read.
