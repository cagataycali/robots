### Fixed: a `stop_when` base predicate that can never fire is refused, not run to budget

`run_policy(stop_when=...)` probes the entities a clause references against the
live scene before arming it, so a typo is an up-front error rather than a rollout
that burns its whole step budget reporting `stopped_reason="budget"`. The walker
collected entities by kwarg, and the `base_*` family's `robot` kwarg was in no
roster - so all 11 base predicates escaped the probe. Two clauses that can never
fire were armed and run: one naming a robot the scene does not have, and one
arming a base term on a robot with no floating base (a fixed-base arm reports
neither `base_pos` nor `base_quat`, which every `base_*` term reads).

The base family is now collected by predicate rather than by kwarg - `robot`
defaults to the sole robot, so a clause references a base whether or not it
spells one - and probed with the new `can_resolve_base`, which reads the same
lookup the predicates use at evaluation time. `predicate_reads_robot_base`
derives the family from the factory signatures, so a predicate added later is
covered without editing a list. Clauses on a robot that does have a floating
base are unaffected.
