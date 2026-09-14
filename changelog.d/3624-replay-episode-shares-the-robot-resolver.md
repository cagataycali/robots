### Fixed: replay_episode resolves robot_name the way every other policy surface does

`PolicyRunner.replay` carried its own copy of the robot resolution that
`run_policy`, `eval_policy` and `evaluate_benchmark` share, and the copy read
`robot_name or <first robot>` where the shared resolver asks whether a name was
supplied at all. An explicit `robot_name=""` - the shape an unset config value
arrives in - was therefore read as "omitted" and a robot substituted in both
scene shapes, and `robot_name=None` in a scene holding several robots replayed
the recorded actions onto whichever one `list_robots()` put first, reporting a
success that named it. Replay is the one policy surface that drives the
actuators from a recording rather than a policy, so the substitution the
siblings refuse was a robot the caller never chose being moved.

`replay_episode` now resolves through the shared resolver, so a supplied name is
never re-resolved and reaches the existing membership check, which reports it by
name with the loaded set, and an ambiguous scene returns the shared
candidate-listing message rather than a replay. `robot_name=None` in a
sole-robot scene is unchanged. The replay docstrings and `docs/recording.md`
described the removed copy ("defaults to the first robot") and now state the
shared rule.
