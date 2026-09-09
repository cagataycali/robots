### Added: the merge-base overlap check reads the open set against a path list before any commit exists

`scripts/check_merge_base_overlap.py --paths <paths>` reports which open
non-draft pull requests already edit the files a change is about to touch.
`--all-open` compares two pushed branches, so it cannot run until the second
exists - and every duplicate pair it has caught opened inside one ~35-minute
window, with both implementations already written. #3368 and #3370 both fixed
the posture-flag read in `strands_robots/simulation/base.py`; the first was open
and approved for 31 minutes before the second's first commit, every duplicate
key read clean (`Closes` against `Refs`, fragments led by the issue number
against the PR number, no shared edited pre-existing test), and the pair spent a
second external approval on a diff `main` already had. The file a defect lives
in is the one thing two fixes of it cannot avoid sharing, and it is known before
a branch is. The intake mode reads only the open set's path sets - no compare,
because there is no head to compare - blocks on a behaviour-bearing hit, lists a
prose-only one without blocking, excludes drafts and names an unreadable path set
rather than reporting it clear, on the same rules as the sweep. `AGENTS.md` step
1 now runs it beside the duplicate-claim read and no longer says the path
relation cannot be asked at intake.

Refs #3169.
