### Fixed: the node-ID decode AGENTS.md prescribes before a mutation is a reject, never a pass

Step 8 presented the offline node-ID decode as making a mutation's target
"checkable before the write". It is not: GitHub routes an owned object by its
own id, the third field of the payload, and neither validates nor uses the
repository field beside it. A `mergePullRequest` was aimed at
`PR_kwDORUMiZs7Kw3fA`, which decodes to `[0, 1162027622, 3401807808]` - this
repository's `databaseId` in the middle - and resolves to a merged pull request
in `uutils/coreutils`. The prescribed check clears it, and permissions rather
than the check are what stopped the merge.

The decode is kept as the fast reject it soundly is - a middle field naming
another repository is proof of a wrong ID, which is the shape #1916 had - and
the rule is now the one with no decode in it: resolve every ID from a query
naming the object by owner, name and number. The passage also states the
asymmetry that decides how much the rule is worth, since a refused merge leaves
nothing behind while a `createIssue` against a wrong ID succeeds and cannot be
undone by the account that made it, and names the remedy for one that has
already landed. That has now happened twice: `Ali111q/todo#1` was filed twenty
minutes after the problem was reported, from an ID whose repository field reads
`1060491130` rather than this repository's `1162027622` - so the surviving
reject direction covers it, and the two strays name different repositories
rather than one stale value. The passage also scopes the rule to mutations,
since a query names its subject by owner, name and number and cannot address
the wrong repository at all.

`tests/test_graphql_node_id_targeting.py` executes the limit on the recorded
cross-repository ID rather than asserting the prose says so, and keeps the
reject direction pinned so the narrowing is not satisfiable by deleting the
decode.
