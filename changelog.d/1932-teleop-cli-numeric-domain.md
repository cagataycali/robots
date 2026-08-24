### Fixed: the teleop tool refuses a numeric option the lerobot CLI cannot honor

`build_lerobot_command` validated only *names* - a missing `dataset_repo_id`, an
unknown `dagger_input_device` - and interpolated every numeric option into the
argv with a bare `str()`. That argv is handed to `subprocess.Popen(...,
start_new_session=True)`, so the detached process is not a channel the call can
read a failure back from: the session starts, `status="success"` is returned with
a pid, and a value the CLI cannot parse surfaces minutes later in the session's
log file. Measured on a `lerobot-record` argv, `dataset_fps` alone reached
`--dataset.fps` as `0`, `-5`, `2.7`, `nan`, `inf`, `True`, `'30'`, `None` and
`[30]`; `dataset_num_episodes=0` asked for a recording of no episodes,
`dataset_episode_time_s=0` for episodes of no length, and `replay_episode=-1`
put a negative episode index on a replay.

Two options were read for truthiness rather than presence, which made `0` mean
the *opposite* of what it says. `teleop_time_s=0` - the one value meaning "stop
at once" - emitted no `--teleop_time_s` at all, leaving an unbounded teleop
session; a replay with `dataset_fps=0` dropped the rate flag, so the caller's
requested rate was silently replaced by lerobot's own default.

Every option the requested mode emits is now checked against the shared scalar
domain for its kind - the same `strands_robots.utils` domains every other
recording surface in the tree already applies - before the argv is built, so no
subprocess is launched for a call that cannot be honored. The domains follow
lerobot's own config dataclasses: `fps` and the episode counts are declared
`int`, so an integral float read from a config is accepted and emitted as a whole
number (`30.0` -> `30`, which is what the CLI parses into an `int` field), while
`teleop_time_s` is lerobot's `float | None` budget and may be fractional. The
floor is per option rather than uniform: `dataset_reset_time_s=0` (no operator
pause between episodes) and `replay_episode=0` (the first episode) are real
requests and stay accepted, while a zero rate, a zero-length episode and a
zero-episode recording are refused. Only the options a mode actually puts on the
argv are checked, so a caller is never refused for a value the requested mode
ignores.
