### Fixed: `lerobot_camera` refuses a posture it cannot read, instead of taking the opposite one

`async_mode`, `warmup` and `save_config` select a posture rather than scaling a
quantity, and were read by truthiness beside the module's tabled numeric options
and vocabularies. Every non-empty string is truthy, so the words a caller reaches
for when opting out selected the affirmative posture: `save_config="false"` wrote
the configuration file under `status="success"`; `warmup="false"` was handed to
`Camera.connect` as the string, persisted into that file as `"warmup": "false"` -
the one field of that document declared a boolean - and reported on the line above
it as `Warmup: on`; `async_mode="false"` selected the asynchronous read path,
which the plain `False` does not.

The flag also gates a tabled numeric row: `timeout_ms` is only refused under
`async_mode`, because the synchronous read consumes no budget. Reading that gate
by truthiness switched the row off from outside its own table, so `async_mode=0`
with `timeout_ms=-5` was answered `status="success"` - an unusable budget accepted
because a falsy value that is not a declared spelling of *off* discarded the row.

The three flags now go through `strands_robots.utils.boolean_flag_error`, the
domain the sibling `lerobot_train` and `lerobot_calibrate` builders already
consult, keyed by action so `discover` and `list` - which consume none of them -
still refuse none of them. The check runs ahead of the numeric guard, so the
refusal names the flag rather than the budget it gates.
