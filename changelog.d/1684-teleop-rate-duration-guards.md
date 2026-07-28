### Fixed: the teleop loop refuses a rate or a horizon it cannot honor

`teleoperate(hz=..., duration=...)` reads both knobs only inside the control
loop, which runs on a background thread, so an unusable value was reported as a
started session and only misbehaved afterwards:

- `hz=0`, a negative rate, `nan` or `inf` all left the loop period at `0`
  (`period = 1.0 / hz if hz > 0 else 0.0`, and `1 / inf` is `0`). The loop then
  spun as fast as the host allowed - measured at ~25 kHz against a 50 Hz
  request - polling the leader and writing the follower on every pass, while the
  call returned `status="success"` announcing "0Hz" / "nanHz".
- `duration=0` was read by truthiness, so the one value that most plainly means
  "stop now" meant "never stop"; `nan` did the same, and a negative duration
  reported a started session whose loop had already exited with zero frames.
- A non-numeric `hz` raised `ValueError: Unknown format code 'f' for object of
  type 'str'` out of the success message, past the tool envelope, and a
  non-numeric `duration` killed the loop thread while the session still reported
  `running`.

Both are now validated at the call, before any teleoperator is connected and
before any mesh publisher is started, so a rejected call has no side effects. The
same rate knob reaches the mesh publish loop through `start_teleop_publish` and
`InputPublisher`, where `1 / hz` raised (or spun) on a background thread the same
way; both now refuse it too - the publisher at construction, so a bad rate cannot
produce a publisher that reports `running` while publishing nothing.

The accepted domain - any positive finite real, `bool` rejected as an `int`
subclass that would act as a silent `1` - is `positive_finite_number_error` in
`strands_robots.utils`, now shared with the rollout knobs it duplicated
(`run_policy(duration=...)`, `control_frequency`), so the rate contract cannot
diverge between the surfaces that all divide by it. Error text is unchanged for
those existing callers.
