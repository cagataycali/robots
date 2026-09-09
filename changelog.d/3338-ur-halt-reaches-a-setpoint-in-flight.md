### Fixed: a UR halt reaches a setpoint already inside `send_action`

`URDriver`'s rollout loop re-reads its stop event after the policy returns, and
that re-read is what stops the *next* step. It is not the last thing before the
write: the loop hands the action to `send_action`, which reads both mode
registers and the measured pose - three RTDE round trips to the same controller
the halt is talking to - before it calls `servoJ`. A halt issued while those
reads were in flight was answered by one more setpoint, on all three halt verbs.

Measured over the RTDE double, with the rollout parked in the mode gate:

| halt verb | order the controller heard | setpoint after the halt |
| --- | --- | --- |
| `stop()` | `servoStop`, `servoJ` | yes |
| `stop_task()` | `servoStop`, `servoJ` | yes |
| `cleanup()` | `servoStop`, `disconnect`, `servoJ` | yes |

The last row is the one `cleanup`'s own docstring ruled out: clearing the
interface handles under the lock turns away a thread that has not read them, and
a thread inside `send_action` is holding them. `stop_task` reported
`stopped=False` throughout, so the arm moved after a halt the envelope had
already declined to claim.

Every halt verb now bumps a counter before it issues `servoStop`, and
`send_action` re-reads it immediately before the write - the position
`G1Driver` and `Go2Driver` already hold, which they get for free by building and
publishing the frame inline. A counter rather than a flag, so only a write whose
gates began before the halt is refused and a setpoint issued after one is still
written. The dropped setpoint is reported: the loop exits `refused` with a reason
naming the halt, rather than discarding it silently.
