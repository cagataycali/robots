### Fixed: the hardware control loop refuses a rate it cannot honor

`Robot(..., mode="real", control_frequency=...)` computed the loop's per-action
period as `1 / control_frequency` with no validation. That period is the only
throttle between two `send_action` calls on a physical servo bus, and
`asyncio.sleep` returns immediately from any value `<= 0`, so a rate that
cannot be honored did not fail - it removed the throttle:

- `control_frequency=-30` (period `-0.033 s`) and `control_frequency=inf`
  (period `0.0 s`) both left the loop free-running: over one 0.4 s task they
  applied 4784 and 7704 servo commands (about 12 kHz and 19 kHz) where the
  requested rate allows 20, and both reported `status=completed`.
- `control_frequency=nan` made the period `nan`, which `asyncio.sleep` refuses:
  the task failed with `Invalid delay: NaN (not a number)` *after* the first
  action had already been applied to the arm.
- `control_frequency=True` silently ran a 1 Hz loop (`bool` is an `int`
  subclass), and `0` / `"30"` surfaced a bare `ZeroDivisionError` /
  `TypeError` from the constructor.

The rate is now validated before the period is computed, and before
`_initialize_robot` opens the serial port - so a rejected rate never reaches
the arm. The accepted domain (any positive finite real, `bool` rejected) is the
one the simulation already applies to the same knob in
`SimEngine._validate_positive_frequency`, and a test asserts the two agree
value by value, so a rollout cannot be refused for a digital twin and accepted
for the arm it mirrors.
