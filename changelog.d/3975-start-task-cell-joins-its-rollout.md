### Tests: the `start_task` admission cell joins the rollout it submits

`test_hardware_after_shutdown.py::test_start_task_before_cleanup_still_submits`
submitted a real task with the default `groot` provider on port 5555 and
returned, leaving the fixture's `cleanup()` to race the worker's bring-up. When
the worker won, it built a `Gr00tPolicy` that dialed `tcp://localhost:5555`
for a server no test provides and the teardown waited out the client's 15 s
`reset` budget - 15.0 s on one CI run against 0.06 s when the teardown won. The
cell now stubs the policy build with the file's counting double, gives the task
a 50 ms budget and joins its future before teardown, and asserts the rollout it
submitted connected the arm, reset the policy and commanded an action, rather
than reading the "Task started" text alone (refs #3869).
