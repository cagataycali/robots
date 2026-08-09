### Fixed: an execution-posture flag `lerobot_teleoperate` cannot honor is refused instead of read by truthiness

`lerobot_teleoperate`'s `background` and `auto_accept_calibration` choose how the
tool *runs* the command it built, rather than what appears on the argv, so the
per-mode flag table that covers the builder's flags has no entry to scope them
by. Both were read as `if <flag>:`, and every non-empty string is truthy, so the
words an operator reaches for when opting out selected the affirmative posture:
`background="false"` detached the session instead of running the foreground one
that was asked for, and `auto_accept_calibration="false"` still wrote two
newlines into the child's stdin, accepting whatever calibration prompt the robot
was showing. `None`, `[]` and `0` took the negative branch just as silently,
without ever being a declared spelling of it. Both are reachable from an agent -
the tool spec declares each `{"type": "boolean", "default": true}`.

`auto_accept_calibration` is the sharper of the two: the posture `background`
reaches is at least reported back, since the envelope carries `"background"`, a
pid and a log file, while nothing anywhere reports that stdin was written to.

Both are now checked against the shared `boolean_flag_error` domain at the top of
the `start` / `dagger` branch, so a refused call builds no argv, records no
session and starts no process, and the refusal reads identically to the one the
builder's own flags already give. The check is scoped to that branch because no
other action reads either flag. `auto_accept_calibration` and
`dagger_record_autonomous` also gain the tool-spec descriptions they were missing
- a model deciding whether to withhold a calibration was reading the generated
placeholder `"Parameter auto_accept_calibration"`.
