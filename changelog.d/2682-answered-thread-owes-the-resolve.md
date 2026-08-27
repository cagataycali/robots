### Fixed: an answered review thread is now reported as owing its resolve

`check_thread_is_answered.py` classified a thread whose last non-bot comment is the
author's as `answered` and called it "not work". That is true of the *reply* and
false of the merge: `answered` is chosen only after `isResolved` has been ruled out,
so every thread carrying it is answered **and** unresolved, and
`required_review_thread_resolution` is a branch ruleset rule. Measured on #2680 at
head `0fd0f4e3`, minutes apart with the same token: this sweep reported
`nothing-owed` and exited 0, while `check_merge_blockers.py` reported
`unresolved-threads` owed by the author. A sweep whose stated contract is "which of
my open pull requests actually need me" cleared a pull request that could not merge
until its author acted, and the cost was a whole scheduled cycle spent reaching the
wrong conclusion.

An answered-but-unresolved thread is now author-owed work: it exits 1, and the
per-pull-request outcome is `author-owes-a-resolve` (or `author-owes-a-reply`, which
outranks it, since a thread still owed an answer is not yet ready to be resolved).
The remedy printed for it is the resolve **and an explicit refusal to reply** --
restating a landed fix is the failure the check was written to prevent, and making
the resolve owed must not license it again. `settled` is now the only outcome that
is not work. AGENTS.md step 5 gains the matching fourth bullet, so an agent
reasoning from the file reaches the same answer as one running the command.
