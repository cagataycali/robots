### Docs: the apt step's comments and the bound grader's message point at the job bound, not at a number it no longer holds

#2457 raised `test-lint.yml`'s job bound 45 -> 60, and three places kept saying
45 in the present tense: two comments on the apt step ("rather than at the
job's 45", "into the 45-minute reap") and the failure message of
`test_the_bound_is_tighter_than_the_default`, which read "the widest job in
this tree is the 45-minute suite" beside a `_CEILING_MINUTES` of 60. The
comments now point at the `timeout-minutes` line above them, the sizing
arithmetic that held at 45 is marked as what it was measured against, and the
message derives its number from the suite job's declared bound through a
`_suite_job()` helper that also replaces two duplicated lookups. No bound
changes; the decision #3143 asks for is untouched (#3340).
