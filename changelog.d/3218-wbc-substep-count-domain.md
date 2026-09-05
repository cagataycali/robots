### Fixed: `WBCTorqueController`'s substep count is the count the caller named

`WBCTorqueController` declares `owns_stepping`, so
`physics_substeps_per_control` is not a hint -- it is the number of `mj_step`
calls one `apply()` makes, and therefore the control period itself. At the SONIC
0.005 s timestep the upstream `g1_gear_wbc.yaml` cadence of `4` is one inference
per 20 ms (50 Hz).

It arrived through `max(1, int(physics_substeps_per_control))`, which cannot
report an unusable count -- it substitutes a usable one. Measured on the shipped
`unitree_g1` scene, `0`, `-5` and `True` each became `1`, so one action advanced
5 ms instead of 20 ms and the gait ran at 200 Hz where 50 Hz was asked for;
`4.9` was truncated to `4` and `"4"` coerced to it. Every one of those rollouts
completed with nothing naming the count that had been replaced, and
`float("nan")` raised `ValueError: cannot convert float NaN to integer` from the
`int()` itself, naming neither the parameter nor the class.

The count is now held to `positive_whole_number_error`, the shared domain whose
docstring already names "the physics steps one applied action is held for" as
one of its two families, and which every backend's `send_action(n_substeps=)`
uses for the same quantity. `PolicyRunner._control_substeps` had already been
converted away from this exact clamp -- its docstring records that `0`/`-5`
"silently collapsed to a single physics step" -- so this controller was the
un-converted copy of a rule the rest of the tree already applies. An integral
float and a NumPy integer stay first-class, as they are everywhere else in that
domain; `int()` is applied after the guard rather than before it.

Why a substituted count is not merely a slower rollout: the gait phase advances
by the control period the loop actually runs at, so a commanded
`gait_frequency` under a replaced count means something other than steps per
second, and the robot walks at a rhythm nobody commanded while every reported
number still looks right.

The existing pin derived its expected physics advance from the attribute the
controller stored, so it read as correct for any count that had been
substituted; it now derives it from the nominal decimation. The constructor,
previously undocumented, gained an `Args:` block -- the parameter that refuses
is the one a caller most needs documented.
