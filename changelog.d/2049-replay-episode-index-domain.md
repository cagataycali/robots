### Fixed: refuse a replay episode index the dataset cannot be indexed by

`episode` reached `load_lerobot_episode`'s bare `episode < 0` test on
`PolicyRunner.replay`, `SimEngine.replay_episode` and the loader itself, while
`replay_episode` - the same quantity on the lerobot teleop tool, and one of the
two parameters `non_negative_whole_number_error` names in its docstring -
already carried the shared rule. That test gave a verdict to three classes of
value it could not honor: a bool passed it (`True < 0` is False) and then
indexed the episode table as an int, so `replay(episode=True)` resolved and
replayed **episode 1** under `status="success"`; a non-integral or non-finite
index passed it too and was blamed on the dataset (`Episode 2.5 has no frames`)
after a full-length boundary scan; and a str, list or `None` reached the
comparison itself and raised `TypeError`, which is neither the `ValueError` the
loader documents nor the structured envelope `replay` documents. The index now
shares `non_negative_whole_number_error` on all three surfaces and is refused
before the dataset is downloaded, so an unusable index costs no hub round-trip
and no action reaches the actuators. An accepted integral float or NumPy scalar
is coerced with `int()` after the guard has round-tripped it, which also keeps
it on the O(1) `episode_data_index` lookup instead of the O(len(dataset))
last-resort frame scan a float index fell through to.
