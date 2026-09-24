### Fixed: the open-set overlap sweep grades a shared docs page against the word budget

`scripts/check_merge_base_overlap.py --all-open` listed a pair sharing only a
`docs/**` page under "Also sharing a path, not reported (prose-only)", on the
ground that prose cannot change what the suite does and a genuine collision
inside one surfaces as a merge conflict. `tests/test_docs_pages_are_within_the_word_budget.py`
makes both halves false for a page near the budget: two additions to one page
compose to a count neither branch has, and two paragraphs added in different
sections merge without a conflict marker. #3907 and #3940, both approved with
auto-merge armed, both edited `docs/policies/moveit2.md` - base 1479 words, heads
1493 and 1497, each passing, 1511 composed against a budget of 1500. The sweep
reported the pair as prose-only, and whichever squashed second would have turned
`main` red on the required check with nothing to resolve.

For a prose-only pair sharing a `docs/**/*.md` page the sweep now reads the three
blobs - each head's and the base's, all from the base repository, whose object
store holds a fork's pull request commits too, at the shas the compare payload
already carries -
counts words as the grader does (`len(text.split())`, UTF-8), and reports the
pair in its own section, with the three counts and the sum, when the composition
exceeds the budget while each head alone is inside it. That is three requests
per shared page and none for the rest of the queue; the finding sets the exit
status on the same footing as a shared behaviour-bearing path. A page whose sum
stays inside keeps the prose-only exemption, a head already over alone is that
branch's own red rather than a composition (which is also what leaves an
exempted page alone: an exemption is over on every head), and a blob the sweep
could not read is named as unevaluated rather than counted as inside. The budget
is carried as `DOCS_WORD_BUDGET` in the script, because the sweep runs with no
checkout and imports nothing outside the standard library, and pinned equal to
the grader's `_BUDGET` from the grader's own source.
