### Fixed: a predicate-DSL clause whose entity name cannot name anything is refused

`make_predicate` -- the one choke point every clause in the predicate DSL passes
through -- held every NUMERIC kwarg to a domain and left the kwargs that name a
scene entity untouched, which its own docstring stated. Every
`(predicate, name param)` pair the registry declares accepted `""` and compiled
clean, and each one left the clause permanently decided: the arm-time probe
cannot see a blank name either, because the collector feeding it gathers only
non-empty strings, so `body=""` is never handed to `can_resolve_body`.

Twenty of the twenty-one pinned the term to a constant that reads as an honest
miss -- `False` for a bool predicate, `0.0` for a reward term. `grasped`'s
`gripper_prefix` inverted it. It is compared with `str.startswith`, and the empty
string is the identity prefix, so a blank one selected EVERY geom in the scene
instead of the gripper's: on an SO-101 scene with a 25 mm cube resting on the
floor 300 mm from the gripper, `grasped(body="cube", gripper_prefix="")` answered
`True` on the cube's contact with that floor. `run_policy(stop_when=...)`
returned `status="success"`, `stopped_reason="predicate"`, `steps_used=1` with
the cube 0.000 mm from where it was placed, and `evaluate_benchmark` scored
`success_rate: 1.0` over three episodes at `avg_steps: 1.0` under
`success_measured: true` -- a success reported for a rollout that never touched
the object.

A kwarg the factory annotates as a required `str` name must now be a non-empty
string, refused at the same choke point as the numeric domains and so covering
the per-stage calls a `staged_reward` compiles by calling back into it. The
refusal names the predicate, the parameter and the consequence, and states the
opposite consequence for a prefix-matched name than for a whole-compared one.
A non-string name is refused there too, rather than escaping as a bare
`TypeError: startswith first arg must be str` from inside the evaluation loop.
The `base_*` family's optional `robot` selector keeps its `None`, which is the
documented sole-robot default rather than a missing name.
