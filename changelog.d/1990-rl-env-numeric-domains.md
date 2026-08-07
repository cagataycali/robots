### Fixed: an unusable `SimEnv` numeric is refused instead of silently scaling the policy out of the loop

`SimEnv.__init__` validated three of its arguments and silently coerced the rest. `action_scale`
went through a bare `float()`, `max_episode_steps` and `action_dim` through a bare `int()`, and
`n_substeps`' own hand-rolled `int(n_substeps) < 1` comparison had the `bool` hole every such
comparison has.

`action_scale` is the consequential one, because it multiplies every action the env sends. The
constructor already documents this exact pathology one argument away - `n_substeps`' docstring
explains that "the PD controller needs several substeps to track it, so a single substep barely
moves the arm" - and the scale does the same thing more completely by scaling the target itself.
Measured on a MuJoCo two-joint arm over 60 steps of a constant `[0.9, -0.7]` command: `1.0` drives
the shoulder to `+0.5235 rad`; `0` leaves it at `-0.0` (the elbow's `+0.0117` is gravity sag);
`-1.0` inverts it to `-0.4511`; and `nan`/`inf` have `send_action` refuse all 60 commands. Every one
of those runs banked the same full `60.0` return, because `step` discards `send_action`'s status -
verbatim the pathology the `num_actions` comment three lines below in the same constructor
describes ("every step wrote no target while the reward was still collected"), reached through the
scale instead of the width. `True` was a silent scale of `1.0`, and `None` / a list raised a bare
`TypeError` out of a constructor that otherwise raises documented `ValueError`s.

The other two were the same shape: `max_episode_steps` of `0` or below reports a time-out on the
first step, so every episode was over before it began - and reported it as a *truncation*, which
on-policy GAE value-bootstraps; `action_dim=0` sized the action head to zero outputs and `-1` to a
negative width.

All four now read their domain from one table, checked before the engine is touched so a refused env
cannot leave a stepped simulation behind. Each domain is the one its *consumer* can honor, so nothing
is refused here that the code downstream accepts: `positive_finite_number_error` for the continuous
multiplier; `positive_whole_number_error` for `n_substeps`, which is `send_action`'s own domain for
that parameter and is what `SimEnv` forwards it to (the narrower guard would have refused an
`np.int64` or a `3.0` from a config that `send_action` honors) and for `max_episode_steps`, which is
only ever compared; and `positive_count_error` for `action_dim` alone, because it sizes the trainers'
action head where an integral float raises rather than being coerced.

`action_dim=None` keeps its documented meaning, a fractional and an `np.float32` scale are unchanged,
and the hand-rolled `n_substeps` comparison is gone - which also resolves a wording inconsistency an
earlier contract had recorded as deliberate, since `SimEnv` and `send_action` now refuse the same
parameter through the same guard. A structural test derives the set of numerics from the live
signature, so one added later cannot ship without a domain.
