### Changed: the preflight roster derivation parses only the modules that name a walk

`scripts/check_whole_tree_graders.py` derived its roster by parsing every test
module and resolving its imports, helpers, call sites and enclosing scopes in
several passes over the tree - for the two modules in three that call no
`rglob`, `glob`, `iterdir` or `walk` at all. The cell that pays for that
derivation, `test_the_roster_covers_every_grader_the_issues_name`, was among the
largest single cells in the suite (#3869). A module that calls one of those
methods spells its name as a token, so a source with no such token is now
skipped before it is parsed. Measured on this tree: 11.9 s -> 5.7 s locally with
the same 164 graders derived, and `hatch run whole-tree-check` pays the same
derivation. The one direction the skip could fail - a walk whose layout the
token scan misses - is pinned beside it.
