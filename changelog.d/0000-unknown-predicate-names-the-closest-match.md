### Fixed: an unknown predicate name says which one you probably meant

`make_predicate`, `predicate_kind` and `predicate_reads_robot_base` - and so
every `stop_when`, `success_when` and benchmark-spec clause that names a
predicate - refused a typo like `graspd` with only the full list of thirty
names. The refusal now leads with `Did you mean 'grasped'?` when a registered
name is close, the same shape as `Robot("<typo>")`, and still lists every valid
name after it. A name nothing resembles gets no guess.
