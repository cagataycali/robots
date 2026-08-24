### Docs: a clean base merge is free of dismissal and not free of last-push approval

`AGENTS.md` already recorded both halves of this - that refreshing an approved
branch over a fixed `main` is cheap when the merge is conflict-free (#1821), and
that pushing to a contributor's branch consumes the approval of whoever owns the
token (#1722). It did not say that the first is scoped to *dismissal only*, and
the two are easy to read as one rule, because the reassuring measurement and the
hazard live in different paragraphs.

#1035 is the case where both were observed on a single head, so the two
mechanisms separate cleanly. CI triage on it correctly diagnosed a stale merge
base and prescribed `git merge upstream/main`; the refresh was then executed with
the maintainer's token rather than requested from the contributor, and
`8d6a4c42` - `Merge branch 'main' into feat/ackermann-ros-robot`, authored and
committed by the maintainer - became the head. The merge was clean: `git show
--cc` is 0 lines and the PR's own diff is unchanged at `7 files, +900/-26`. The
pre-existing review is therefore **still `APPROVED`**, not `DISMISSED`, exactly
as the #1821 table predicts - and `reviewDecision` is nonetheless
`REVIEW_REQUIRED`, with a second approval from that same account on that exact
head, every check `SUCCESS`, failing to move it.

So `dismiss_stale_reviews_on_push` keys on the PR's own diff and
`require_last_push_approval` keys on the pusher's identity. A clean base merge is
free under the first and not under the second, and no amount of re-approving from
the pushing identity helps. On your own branch the refresh is cheap outright; on
a contributor's it converts a pull request one maintainer could merge into one
that needs a second, with nothing in the PR's own fields saying so.

The control table in `AGENTS.md` step 8 listed #1035 as the row that reads
`APPROVED`. That is now stale as a statement about the current pull request, so
it is split per head - `2be59dad` `APPROVED`, `8d6a4c42` `REVIEW_REQUIRED`, one
input changed - which also makes it a stronger control than two different pull
requests were. A closing note records that the commit metadata cannot settle the
question in either direction: #1722's author and committer are `strands-robots`,
reading as satisfied when it is not, while #1035's head names the maintainer
outright, and only
`GET /repos/{owner}/{repo}/actions/runs?head_sha=<head> -> triggering_actor` is
load-bearing.

Documentation only; no production code or test behaviour changes.
