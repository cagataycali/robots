### Changed: the pull-request triage tools moved to `.github/scripts/`, and their replay suites left `tests/`

Five tools read the GitHub API and report on an open pull request - whether a
claim collides, what blocks a merge, whether a head record is current, whether a
checkout is the head it claims, whether a review thread owes a move. No workflow
runs any of them; an operator or an agent does, on demand. They now sit in
`.github/scripts/` beside `agent_api_snapshot.py`, which is the same kind of
tool, and their unit suites are removed rather than moved:

| removed from `tests/` | lines | cases | what it replayed |
| --- | --- | --- | --- |
| `test_merge_blockers.py` | 1,580 | 111 | the blocker classifier over captured `#1035`, `#2497`, `#2566`, `#2574`, `#2907`, `#3205`, `#3314`, `#3315` payloads |
| `test_duplicate_work_added_path_key.py` | 958 | 95 | the created-path key over the `#1942/#1944`, `#1994/#1995`, `#2007/#2015` pairs |
| `test_thread_is_answered.py` | 520 | 46 | the thread verdicts over `#1722` and `#2577` |
| `test_duplicate_work_described_change_key.py` | 442 | 57 | the described-change key over the pairs the path key misses |
| `test_checkout_is_pr_head.py` | 337 | 35 | the checkout classifier over `#2520`'s four instances |
| `test_pr_head_is_current.py` | 294 | 23 | the head-record classifier over `#2538` |
| `test_push_time_claim_recheck.py` | 248 | 28 | the push-time re-read of the claim question |
| **total** | **4,379** | **395** | |

Measured on `b91f49f4`: those 395 cases cover **0 of the 53,803 statements in
`strands_robots`**. What they pin is the shape of GitHub API responses captured
from specific 2026 pull requests - a snapshot of a third party's payloads, held
in the library's test budget, from which no `strands_robots` behaviour is
reachable. `tests/` goes from 509,463 to 505,103 lines and `scripts/` from
10,198 to 5,793.

Two suites stay, because each also grades a tool that keeps living in `scripts/`
and is run by a workflow step: `test_duplicate_claim_check.py` (a documented
invocation of `check_merge_base_overlap.py` or `check_last_push_approval.py`
names its repository) and `test_last_push_rule_names_the_update_branch_button.py`
(what `check_last_push_approval.py` prints on a finding names the "Update branch"
button). Both now derive their population from **both** tool directories, so
relocating a tool does not quietly drop it from the requirement.

Composition across the new boundary is explicit: `check_duplicate_claim.py`
resolves `scripts/assemble_changelog.py` and `check_merge_blockers.py` resolves
`scripts/check_last_push_approval.py` from the repository root rather than from
beside themselves.
