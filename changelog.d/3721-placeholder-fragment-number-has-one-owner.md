### Fixed: a placeholder fragment number is refused by the checks that run before the suite

`0000-` and `999x-` fragment names were refused only by
`tests/test_changelog_fragments.py`. The rule was written in that test, so
`scripts/assemble_changelog.py --check` -- the command `changelog.d/README.md`
hands a contributor -- printed `changelog fragments OK` for a placeholder name,
and the pull-request convention job, which imports the assembler's
`validate_fragment` precisely so the two cannot disagree, passed it too. A
branch was told locally that the name was fine and learned otherwise from the
required suite, and a branch that merged in between put an entry on the log with
no pointer back to the change that made it.

The rule now has one owner, `assemble_changelog.is_placeholder_number`, applied
in `validate_fragment`, so `--check`, `--apply`, the convention job and the
suite give one verdict. The stray fragment already on the log is renamed to the
pull request that landed it.
