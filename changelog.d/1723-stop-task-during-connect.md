### Fixed: a stop pressed while the robot is still connecting actually stops the task

`Robot.stop_task()` only recognised `TaskStatus.RUNNING`. But
`_execute_task_async` spends the whole hardware bring-up in `CONNECTING` - a
motors-bus handshake plus `warmup_s` per camera, seconds on a real SO-101 and
longer on a multi-camera rig - followed by the policy build. A stop pressed in
that window was answered `status="success"` with `"No task running to stop"`,
and the arm then moved anyway:

```
stop at t=0.9s  status=connecting -> success: No task running to stop (current: connecting)
stop at t=1.2s  status=connecting -> success: No task running to stop (current: connecting)
stop at t=2.0s  status=connecting -> success: No task running to stop (current: connecting)
servo writes after the three stops: 192
final task status: completed
```

Three stop presses reported success, the arm was then driven for a full
rollout, and the task reported itself completed. `mesh/core.py` routes the
fleet `{"action": "stop"}` dispatch straight into `stop_task`, so the fleet
interrupt inherited the same hole.

Signalling through `_task_state.status` cannot express the request on its own:
`_execute_task_async` writes `RUNNING` once bring-up finishes, overwriting any
`STOPPED` recorded before it. The request is now latched in an event set before
the status is even read and cleared only when a new task starts; the rollout
honors it at each stage boundary (after connect, after the policy build), in its
loop condition, and in the terminal block that decides `COMPLETED` vs `STOPPED`.
A stop during bring-up now commands the arm zero times, never initializes the
policy, and is reported as `Task stopped (during connect)`. A stop mid-rollout,
and a stop on an idle or terminal robot, are unchanged.
