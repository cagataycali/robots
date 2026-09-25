### Changed: one `g1_task` verb over the G1's control-loop task lifecycle

`g1_start_task`, `g1_get_task_status` and `g1_stop_task` are one table-driven
`g1_task(driver, action)` over the three methods `G1Driver` exposes for the task
its 500 Hz control loop runs. `action` is `"start"`, `"status"` (the read-only
default) or `"stop"`; the per-action differences live in one table - which driver
method is called, whether the flat envelope carries `stopped`, and whether
presence also requires the snapshot's own `steps` field.

Every value a caller could read is unchanged. `status` and `stop` still flatten
`_ControlLoop.snapshot`'s eleven fields plus `present` and `reason`, still
surface the driver's own `status` verbatim so a join that outlasted its budget
reaches a caller as an error, and still report `present=False` with every field
`None` rather than fabricating a zero for a loop that never ran. `start` still
passes the driver's envelope through unreshaped, so today's registry-not-wired
refusal - and the loop-start envelope that replaces it once the provider
registry lands - reaches a caller the moment the driver writes it.

Callers rename `g1_start_task(driver, ...)` to
`g1_task(driver, action="start", ...)`, `g1_get_task_status(driver)` to
`g1_task(driver)` and `g1_stop_task(driver)` to `g1_task(driver, action="stop")`.

Three modules (628 lines) and three suites (837 lines) become one module (254)
and one suite (490), and the suite gains two rules none of the three had: the
flat envelope's field set is derived from `_ControlLoop.snapshot`'s own dict
literal, so a field the loop gains and the verb drops fails rather than going
unnoticed; and the live-handle guard is graded on all three actions rather than
only on the one each file was written for.
