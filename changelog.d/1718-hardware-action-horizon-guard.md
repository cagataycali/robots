### Fixed: a hardware task refuses an action_horizon the control loop cannot honor

`Robot.__init__` stored `action_horizon` without validating it, and the task loop
hands that value to `resolve_chunk_length`, which coerces it with
`max(int(action_horizon), 1, ...)`. The coercion turned a caller mistake into a
plausible-but-different rollout on the physical arm rather than an error: a `0`
or negative horizon was silently clamped to one action per inference, so an
open-loop chunked checkpoint was re-queried every single step instead of
replaying the chunk it was trained to emit -- exactly the out-of-distribution
operation `resolve_chunk_length` documents as the reason not to shrink the
interval. Measured on a policy emitting an 8-action chunk, `action_horizon=0`
reported `status="success"` after 38 inferences for 38 applied actions where the
default horizon needed 5. `2.7` was truncated to 2, `"4"` string-coerced to 4,
and `True` acted as a silent horizon of 1. A value `int()` cannot convert
(`None`, `nan`, `inf`, a list) reached that coercion only after the arm was
connected and the first observation had been inferred on, aborting the task with
a bare `TypeError`/`ValueError` naming an `int()` internal rather than the
parameter.

Every one of those values is refused by the simulation's rollout counts, so the
same horizon was rejected for a digital twin and accepted for the arm it
mirrors. `action_horizon` now raises `ValueError` at construction, before
`_initialize_robot` opens the serial port, so a rejected horizon never touches
the arm -- matching the `control_frequency` guard alongside it.

The positive-integer count domain moves to
`strands_robots.utils.positive_count_error` so the hardware constructor and
`SimEngine._validate_positive_int` share one implementation across a layer
boundary `hardware_robot` cannot cross by importing `simulation`. That also
closes a `bool` hole in the shared guard: because `bool` is an `int` subclass, a
bare `value < 1` test rejected `False` while letting `True` through as a silent
count of 1, so `n_episodes=True`, `max_steps=True` and `action_horizon=True` were
accepted as 1. Two callers had worked around the hole locally rather than the
guard enforcing it -- `_validate_control_substeps` rejected `bool` itself before
delegating, and the `run_policy` agent tool still rejects it for its own required
`n_episodes` -- and the redundant local branch is now removed. Every existing
error message is unchanged.
