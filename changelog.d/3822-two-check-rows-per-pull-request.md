### CI: two check rows per pull request, and the guards fail before the install

Every pull request started twelve workflow runs - the required check and eleven
advisory rows, each with its own checkout and `setup-python` for a script that
finishes in seconds (13 rows and 7.1 runner-minutes beside the suite on #3816),
and one of them ended with a step named "Always pass". Now two rows: `ci.yml`
keeps the required `call-test-lint / Test and Lint` context unchanged and adds a
`security` job (CodeQL and the dependency review in one). Inside the required
check a `Guards` step runs before the install: `scripts/ci_guards.py` calls the
closing-reference, changelog-fragment, merge-base-overlap and `uv lock --check`
scripts with the same arguments and folds their exit statuses, and the
LLM-input scan now fails on a finding instead of annotating one. The public-API
and AgentTool-contract drift reports write to the step summary only; the bot
comments are gone. The docs build runs only when `docs/` or `mkdocs.yml`
changed. Ten workflow files are deleted, the last-push-approval report among
them - the ruleset enforces it, and a check that cannot fail is a row that says
nothing.
