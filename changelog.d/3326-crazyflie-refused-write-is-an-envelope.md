### Bug Fixes

- **drivers/crazyflie**: a flight command the radio would not carry is now refused
  rather than raised. Every caller-facing verb (`send_action`, `set_twist`,
  `takeoff`, `land`, `emergency_stop`) documents "a success envelope, or an error
  envelope naming the refusal" and each ends in a CRTP write, which is where a link
  that opened and then stopped answering shows up - the handle is still live,
  `is_connected` still reads True, and only the write finds out. The write had no
  handler, so the SDK's exception left the verb instead of its envelope. The
  refusal the setpoint repeater already defers to `send_action` - "the next
  `send_action` reports the refusal through an envelope a caller can read, which a
  background thread cannot" - now has a receiver, and so do `stop_task`, the agent
  tool's `land` action and `cleanup`, all three of which were already written for
  it. `cleanup` is the one that lost state: its documented "every step tolerates a
  half-built driver" was defeated by its own first statement, so a refused descent
  skipped the repeater halt, the telemetry block, `close_link` and the state clear,
  leaving the process holding a radio link and reporting itself connected. The
  raise set moves to one module constant, `LINK_WRITE_ERRORS`, so the repeater and
  the verbs cannot drift apart; a refused `emergency_stop` says the motors were
  **not** cut and points at a hardware cutoff; and a refused setpoint restores what
  was latched before, because that setpoint is what feeds the firmware supervisor.
