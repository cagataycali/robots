### Fixed: a hardware task's `policy_port` is judged before the arm is connected

`Robot.start_task` and the `execute` chokepoint validated `duration` and
`n_steps` before claiming the motors bus, but read `policy_port` only inside
the async task body - after `_connect_robot` had energized the arm. A port no
policy could ever be built from therefore spent the whole bring-up window (a
motors-bus handshake plus per-camera warmup, seconds on a real arm) before
failing, and `start_task` reported `status="success"` and "Task started" for
it because the failure surfaced on the executor thread. A supplied-but-falsy
`0` / `False` was reported as `"policy_port is required"`, telling the caller a
port they had passed was missing.

Both entry points now check it beside the budget, on the shared
`tcp_port_error` domain the policy providers already apply, so the same port
cannot be accepted by the arm's task entry points and refused by the provider
they hand it to. A refused port connects nothing and leaves the command bus
free. `_execute_task_sync` skips the check when a pre-built `policy_object` is
given, because the port is not read on that path.
