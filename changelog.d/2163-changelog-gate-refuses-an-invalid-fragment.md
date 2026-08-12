### Fixed: the changelog gate now refuses a fragment the assembler would refuse

`Changelog Fragment Check` asked only whether an entry had been written into
`CHANGELOG.md` instead of a `changelog.d/` fragment. It said nothing about whether
the fragment that *was* written is valid, so a malformed or misnamed fragment
reached only `tests/test_changelog_fragments.py`, inside the required suite: on one
pull request whose sole defect was a fragment first line reading `Fixed: ...`
instead of `### Fixed: ...`, this job reported success in 10 seconds while the
required suite reported failure 20.8 minutes later, worded as a test failure.

It now also validates the fragments a branch adds or modifies, using
`assemble_changelog.py`'s own naming and heading rules rather than a second copy of
either, and annotates the offending file directly. Only the branch's own fragments
are read, so a fragment already on the base branch is never attributed to it, and
the remedy stays self-clearing: add the `### `, or rename the file.
