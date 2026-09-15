### Docs: the twelve dashboard passkey-auth variables are documented, and a key built from a family prefix is graded

`strands_robots.dashboard.auth` binds `_ENV = "STRANDS_DASH_AUTH_"` once and
reads each member as `os.getenv(_ENV + "ENABLED")`, or through a resolver
spelling `var = _ENV + name`. Neither shape puts a whole name anywhere
`tests/test_env_vars_the_package_reads_are_documented.py` could see it - the
constant holds a prefix and the literal a suffix - so twelve variables were
read and no page under `README.md` or `docs/` named one, among them the
`ORIGIN` and `RP_ID` the module's own refusals tell an operator to set, and
the durations and challenge caps it refuses rather than defaults.

`docs/reference/configuration.md` gains a block naming all twelve with their
defaults and guards, and the grader now concatenates a `+` chain of literals
and module constants - including through a resolver that wraps the parameter
it reads - so the next family-prefixed variable is graded on arrival.
