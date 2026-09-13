### Fixed: `gr00t_inference` checks its six posture flags instead of reading them by truthiness

`gr00t_inference` tabled its numeric options and its selector vocabularies per
action, but read `use_tensorrt`, `http_server`, `use_sim_policy_wrapper`,
`deterministic`, `remove_volumes` and `force` raw. Every non-empty string is
truthy, so the spellings a caller reaches for when opting out selected the
affirmative posture: `remove_volumes="false"` under `lifecycle="teardown"`
reached `docker rm` as a truthy `-v` and discarded the checkpoints `False` is
documented to preserve; `force="false"` selected the rebuild the caller had
declined; `http_server="false"` moved the port to 8000 and started the REST
server for a caller who asked for ZMQ; and `use_tensorrt="false"` switched the
three dtype rows of the enumerable guard on while `use_tensorrt=0` switched them
off, carrying an unparseable dtype into the detached argv. The
`deterministic` / `n1.7` gate branches on the flag, so `deterministic="false"`
on a legacy protocol was refused as `deterministic=True requires
protocol='n1.7'` - describing the posture the caller had not asked for.

The six are now held to the shared `boolean_flag_error` domain through a
per-action roster mirroring the numeric and enumerable tables, checked ahead of
the protocol gate and the enumerable guard so neither branches on a value it
cannot read. Only the flags an action's handlers are passed are checked, so
`status`, `stop`, `list` and `find_containers` refuse none of them, and
`use_sim_policy_wrapper` is left alone on the legacy protocols that never emit
it. The protocol gate is scoped to the same roster, so an action that never
mounts the wrapper is no longer refused for it. Pinned by
`tests/tools/test_gr00t_posture_flag_domain.py`, whose roster is read from the
tool's signature so a seventh flag cannot skip the domain.
