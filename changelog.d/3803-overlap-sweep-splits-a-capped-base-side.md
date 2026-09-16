### Fixed: the open-set overlap sweep grades a stale base too wide to fetch whole

`scripts/check_merge_base_overlap.py --all-open` read each pull request's base
side - the paths that landed on `main` since the branch forked - from one compare
call, whose `files` list GitHub caps at 300. A capped list is indistinguishable
from a complete one in the payload, so the sweep named those pull requests
unevaluated rather than intersecting a set it knew was short. That is the right
call over a wrong answer and the wrong one over an answer, because the width is
not incidental: `M..base` grows for as long as a branch sits in review, so the
ranges that reach the cap are exactly the longest-standing stale bases, and a
stale base under a path the branch edits is the composition that put `main` red
at `0e636f8` arriving from the base instead of from a sibling. On the open set
of 29 it declined 10 pull requests, every one of them a stale base.

`files` is capped where the payload's `commits` is not, so a range too wide to
read whole still carries a boundary to split it at. A capped compare is now
re-read as `base...boundary` and `boundary...head`, each through the same path
and so split again if it is capped too; each half is strictly shorter than the
range it came from, so this terminates. The union is never short of the capped
list, which is the direction that matters here - it can be longer, since a path
one commit in the range created and another removed nets out of `M..base` and
survives in a half, and `main` did touch that path. Checked against `git diff`
over five capped pull requests on the live queue at 380-780 paths each: no path
missing in any of them, at 5-17 requests.

The floor keeps the old refusal, and names which floor it hit: one commit whose
own diff reaches the cap has nothing left to split on. The pairwise mode is
unaffected either way - it reads the head side, and an unreadable base side has
never removed a pull request from it.
