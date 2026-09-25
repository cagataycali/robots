### Fixed: `whole-tree-check` rosters a grader whose walk root is a local bound from a loop variable

`scripts/check_whole_tree_graders.py` derives its roster by resolving each
walk's receiver to a path, and it already read two spellings on their own: a
root bound inside the walking function (`root = ROOT / "strands_robots"`) and an
area held in a loop variable (`(ROOT / tree).rglob(...)` for `tree` in a literal
tuple). Their composition - `for name in _SCAN_ROOTS: root = _REPO_ROOT / name;
root.rglob(...)` - resolved under neither: the local resolver cannot see through
a loop variable, and the loop resolver reads `base / <loop variable>` only when
that expression is the receiver itself. Three whole-tree graders were unrostered
on the spelling (`tests/test_codeql_query_filters.py`,
`tests/test_documented_guard_names_resolve.py`,
`tests/test_source_no_implicit_string_concatenation_in_lists.py`), and the third
took an approved pull request red behind a green preflight (#4043, #4044). The
derivation now reads such a receiver as the expression the enclosing functions
bind it to and hands that to the loop resolver unchanged, so membership is still
decided by `walk_targets`; the roster moves from 169 to 172 at the same tree.
Pinned by `tests/test_whole_tree_graders_roster_is_complete.py`, with controls
holding a subpackage-per-backend loop, a run-time iterable and a local in an
unrelated function unselected.

Closes #4044.
