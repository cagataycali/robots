### Fixed: a horizon `PolicyRunner.run` cannot run is refused, instead of selecting the other knob's value

`run` takes the rollout length as two knobs - an explicit `n_steps` step count,
or `duration` seconds turned into `int(duration * control_frequency)` steps - and
picked between them with `if n_steps is not None and n_steps > 0`. That `> 0`
meant a step count outside its domain did not fail: it silently handed the
horizon to the *other* knob, whose default is `10.0` seconds. `n_steps=0`, `-5`
and `nan` each ran 500 control steps and 500 applied actions at 50 Hz - not a
clamp to 1, and not the value the caller typed, but a horizon from a parameter
they never set - while `2.7` ran 2 and `True` ran 1. A non-positive `duration`
returned `status="success"` with zero steps and `stopped_reason="budget"`, the
field a caller reads to decide whether to retry, having applied no action at
all; and `nan` / `inf` / a string / `None` on either knob leaked a bare
conversion or operand error naming neither the parameter nor the method.

Both knobs now carry the domain their entry point already applies, raised rather
than returned because `PolicyRunner` is documented as drivable directly and a
direct caller has no envelope to read a refusal from - the same guarantee the
sibling `action_horizon`, `control_substeps`, `control_frequency`, `seed`,
`rtc_inference_timeout_s` and `max_onframe_failures` knobs of that signature
already provide. `n_steps` is judged whenever it is given, which is the exact
condition `SimEngine._resolve_horizon` judges it on, so a step count refused for
a rollout through the facade can no longer be accepted for the same rollout
driven directly; `duration` is judged only when no step count was given, because
that is the only case in which it sets the horizon. The refusal lands before the
first inference, before any applied action, and before the process-global reseed
`seed=` performs.
