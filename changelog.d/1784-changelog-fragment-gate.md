### Docs: the changelog fragment rule is enforced instead of only documented

`changelog.d/README.md` and `AGENTS.md` both state that a behavioural change
records itself as a `changelog.d/<number>-<slug>.md` fragment and never appends
to `## [Unreleased]` in `CHANGELOG.md`. The reason is measured rather than
stylistic: every branch appends at the same anchor, so two open at once conflict
on ordering alone -- never on meaning -- and because a push dismisses a stale
approval, clearing that conflict costs a full re-approval round per affected
branch.

Sixty-seven fragments and every recently merged pull request follow the rule.
Nothing checked it. With three entries inserted directly beneath the anchor:

```
$ python scripts/assemble_changelog.py --check
changelog fragments OK (66 pending)                      # exit 0
$ pytest tests/test_changelog_format.py tests/test_changelog_fragments.py
30 passed
```

Neither suite can see it. One checks the log's version-heading structure, the
other the contents of the fragment directory, and a fragment that was never
written leaves nothing for either to inspect -- so the one rule the convention
rests on was the one rule with no gate behind it, and a branch could reach
`APPROVED` / `SUCCESS` / `CLEAN` having ignored it.

`scripts/check_changelog_fragment.py` compares the `### ` entry headings under
`[Unreleased]` at the merge base with the same set at the branch head, and names
any the branch added. It is a base diff and not a test because `[Unreleased]` on
`main` already carries 168 entries from before the convention existed, so no
static assertion about that section's contents can hold: what is wrong is the act
of adding to it, and an act is only visible between two commits.

The release path stays open, and exactly. `assemble_changelog.py --apply` renders
a fragment verbatim and deletes the fragment it consumed, so every entry a
release adds is matched to a `changelog.d/*.md` file deleted in the same diff.
That is checked per entry, so a release that also hand-writes an extra entry is
still refused and only the extra one is named. Editing, rewording, reordering, or
retiring an entry already in the log adds no heading and is silent.

Like the merge-base overlap check, the remedy is self-clearing: moving the entry
into a fragment removes the addition from the diff, so doing what the check asks
makes it pass and there is no bypass switch to add.
