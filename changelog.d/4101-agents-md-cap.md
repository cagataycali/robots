### Docs: AGENTS.md is capped at 30 KB

`AGENTS.md` went from 216 KB to 19 KB. The review learnings and the step-8
merge-gate runbook are still in git history (`git show 9f011d3e0:AGENTS.md`).
194 tests in ten modules and 40 more in six others that only checked the wording of those
passages are removed with them; `tests/test_agents_md_is_within_budget.py` now
holds the cap.
