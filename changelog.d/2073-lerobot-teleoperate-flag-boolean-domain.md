### Fixed: a boolean flag the lerobot argv cannot honor is refused instead of read by truthiness

`build_lerobot_command` read all five of its boolean flags for truthiness. Each
selects a *posture* rather than scaling a quantity - `dataset_video` picks the
literal `"true"` or `"false"` on the argv, `record_resume` decides whether
`--resume true` appears at all - and every non-empty string is truthy, so the
words an operator reaches for when opting out selected the **opposite** posture
from the one they read as:

- `dataset_push_to_hub="false"` emitted `--dataset.push_to_hub true`, so a
  detached, unattended recording uploaded its dataset to the Hub;
- `record_resume="false"` emitted `--resume true`, appending into an existing
  dataset - preserving its already-stamped `repo_id` - instead of creating the
  fresh one that was asked for;
- `dagger_record_autonomous="off"` emitted `--strategy.record_autonomous true`,
  recording autonomous rollout episodes into a corrections dataset;
- `display_data="false"` emitted `--display_data true`.

`None` and `[]` took the other branch just as silently, without ever being a
declared spelling of it. Nothing reported any of these: the argv goes to a
subprocess launched with `start_new_session=True`, the call returns
`status="success"` with a pid, and the lerobot CLI parses every one of those
argvs without complaint - it is simply told the opposite posture.

Each flag is now checked against the shared `boolean_flag_error` domain the mesh
provisioning entry points already apply, and only for the flags the requested
mode actually emits: the scoping comes from a per-mode table beside the numeric
one, so `replay` - which emits no flag - refuses none, and a flag a mode never
puts on its argv is never a false rejection.
