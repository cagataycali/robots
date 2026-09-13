### Fixed: evaluate_benchmark resolves robot_name the way every other policy surface does

`evaluate_benchmark` carried its own copy of the robot resolution that
`run_policy` and `eval_policy` share, and the copy asked `if not resolved_robot`
where the shared resolver asks whether a name was supplied at all. An explicit
`robot_name=""` - the shape an unset config value arrives in - was therefore
read as "omitted": in a sole-robot scene the benchmark silently evaluated the
only loaded robot and reported a `success_rate` for a robot the caller never
named, which is the fabricated number the surrounding guards exist to prevent.
In a multi-robot scene the same value was refused as `'robot_name' is required`,
naming the one argument that had in fact been passed.

The facade now calls the shared resolver, so a supplied name is never
re-resolved and reaches the existing membership check, which reports it with the
close-match hint and the loaded set. `robot_name=None` is unchanged in a
sole-robot scene; an ambiguous scene now returns the shared candidate-listing
message. The `robot_name` docstring described the removed copy's intent rather
than either behaviour and has been corrected.
