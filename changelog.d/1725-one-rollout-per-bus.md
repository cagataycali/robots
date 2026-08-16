### Fixed: a second rollout can no longer be admitted onto a busy motors bus

`start_task`, `run_policy` and the agent-tool `execute` action each refused a
concurrent task with `if self._task_state.status == TaskStatus.RUNNING`. That
status is only reached after `_connect_robot()` and the policy build, so for the
whole bring-up window - a motors-bus handshake plus per-camera warmup, seconds
on a real arm - every caller read a non-running status and was let through. Two
control loops then drove one bus: with a 1 s `connect()` and two policies
tagging their own commands, the wire carried
`BBBB...BABABABABABABABABAB` with two `send_action` transactions in flight at
once, `connect()` was called twice on the same port, and both calls returned
`status: "success"`. The bus is half-duplex and not thread-safe, so interleaved
transactions give framing errors or a `Goal_Position` write from one policy
landing between the other's read and write - two different intents applied to
one physical arm. Sharing the single `_task_state` slot also let each rollout
overwrite the other's step count and terminal status, so one of them reported
completing steps it never commanded.

Admission is now a claim taken before the bring-up window opens and released
when the rollout ends - including when it errors, raises, or is stopped during
bring-up, so a failed or interrupted task cannot leave the robot refusing every
later one. The check-and-claim runs under a lock, so callers racing at the same
instant cannot both be admitted, and
`start_task` claims on the caller's thread rather than inside its executor job:
it returns before that job begins, so a claim taken there would have reported
"Task started" and only then turned the caller away. The refusal names the
rollout that actually holds the bus, which during bring-up was previously
reported as whichever task ran last.
