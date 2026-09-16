### Quality: a `0000-` changelog fragment is a placeholder and is refused

Ten `changelog.d/0000-*.md` fragments from ten merged PRs reached `main`. The
placeholder guard in `tests/test_changelog_fragments.py` refused only numbers at
or above `9000`, so the `0000` shape a branch writes before its PR number
exists was never looked at. `--apply` deletes each fragment it folds in, so the
pointer from a release-note entry back to its change would have been lost at
release, and descending sort would have put every such entry at the bottom of
the section however recent the change. The ten fragments are renamed to the PR
that landed each, the placeholder rule now has one owner that refuses `0` as
well as the high floor, and `changelog.d/README.md` says so.
