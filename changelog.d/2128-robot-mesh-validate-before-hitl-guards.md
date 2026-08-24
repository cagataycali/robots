### Quality: pin both validate-before-HITL contract guards in `robot_mesh`

`robot_mesh` validates the `send` / `broadcast` command body before raising the
human-in-the-loop interrupt, then each handler re-reads that pre-validated
command from a sentinel and refuses when the sentinel is still unset. Both
guards are an explicit `raise` rather than an `assert` because `assert` is
stripped under `python -O`, and neither had a test: on today's code the pre-pass
always sets its sentinel, so both raise bodies were unreached.

Deleting either guard, or replacing one with the `-O`-strippable `assert` its own
comment warns against, was invisible to all 275 existing `robot_mesh` tests.
Without the guard the handler dispatches the unset sentinel -- `mesh.broadcast(None)`
is issued fleet-wide -- and the audit line that follows raises, so the dispatch
that did happen is never recorded. The `broadcast` comment claimed the loss would
"silently send an unvalidated cmd"; it is corrected to the measured outcome.
