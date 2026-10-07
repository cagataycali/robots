### Tests: the G1 and Go2 parked-policy cells wait a short join budget, and one cell per driver grades the shipped one

Twenty-one cells in `tests/drivers/` park a policy so the control loop outlasts
`_ControlLoop.stop`'s join budget, then read what teardown reports. Each paid
the full shipped two seconds, although a parked policy outlasts any budget.
They now wait a tenth of a second; one cell per driver drives the shipped
budget and asserts the stop waited at least that long, which nothing graded
before. Same covered lines; the three files run in 8 s instead of 46 s.
