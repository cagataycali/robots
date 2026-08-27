### Fixed: a timed drive whose trailing stop was refused reports it

A timed or repeated `drive` owns its own stop: `MobileBaseRobot`, `RosbridgeRobot`
and `AckermannRosRobot` all follow a non-zero command with a single zero one from
`finally`, so the zero goes out even when the main publish raised. Sending it is
not the same as landing it. That zero is a second command over the same
transport, and every shipped graph tool reports a refusal - a declined operator
approval, a rate limit, a transport failure - as an error envelope rather than by
raising, so the refusal was a value that had to be read. Three of the four
drive-owning classes dropped it inside the `finally` and returned the hold's own
success.

Measured against the real `use_rosbridge` gate with an operator who approves the
hold and declines the stop, on a blocklisted `/cmd_vel`: `drive(linear=1.0,
duration=1.0)` answered `status="success"` with "published 2 message(s)", two
messages reached the wire, the last of them `linear.x = 1.0`, and no zero
followed. The robot is left moving at the commanded velocity and the caller is
told the drive succeeded - and the agent-facing tool description promises a timed
command "stops automatically afterwards", so a caller reading `success` never
issues `stop`. The gate prompts once per publish, so approving a hold and
declining its stop is an ordinary pair of operator decisions. `RtpsRobot` and
`RosBridgedRobot` reproduced it through the shared base.

`AckermannRosRobot` already kept the verdict and compared it against the hold's
own result, which is why this is a consolidation rather than a new rule - its
docstring claimed the "same contract as `RosBridgedRobot.drive`", which was the
reverse of the truth. The rule now lives once as
`_mobile_base.failed_halt_error`, beside the drive contract it belongs to, and
the car's message is unchanged. A hold that failed is still the cause, because a
stop it never got to undo is a consequence of it; a single-shot command owes no
stop, so no verdict is read and it latches by contract as before.

The shared safety contract stated the guarantee as fact - "a timed drive cannot
leave a robot with a live velocity latched" - so it now says that the stop's
verdict is read rather than dropped, and `docs/rosbridge-integration.md` says the
same where a reader learns the fleet contract.
