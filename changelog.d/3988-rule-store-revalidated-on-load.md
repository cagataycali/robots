### Fixed: a global-rule store is re-validated on load, as a trace already was

`harness_memory` stores two kinds of memory in one hand-editable directory and
reads both straight into planner context, but only one was held to its own
bounds on the way back. A trace or summary is re-validated on every
`load_trace`, so a store edited outside the tool is refused naming the file and
the remedy. A rule store was not: `load_rules` returned whatever `splitlines`
produced, so a store that had been bulk-appended or hand-edited came back
`success` with a single 1,000,000-character rule (write cap 2,000), with 5,000
rules (cap 1,000), or with ANSI control sequences inside a line the write path
rejects outright -- while `append_rule` on that same 5,000-line store correctly
refused it as full, which is the module reading its own cap in one direction
only.

`HarnessMemory._read_rules`, the single reader behind both `load_rules` and
`append_rule`'s count, now applies `_validate_rule_text` per line and the
`_MAX_RULES_PER_KIND` cap, and refuses naming the file, the line and the bound.
No new constants: the bounds are the ones the write path already enforces, so a
store at either cap still loads, and `append_rule`'s "store full" refusal stays
the one a session reaches by appending.
