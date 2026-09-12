### Fixed

- `DatasetRecorder(strict=...)` is held to `boolean_flag_error`, the domain
  `create` / `resume` / `push_to_hub` / `sync_dataset_to_bucket` already apply to
  their own posture flags. The constructor's flag selects raise-vs-drop on a
  failed dataset write and was read by truthiness, so it inverted in both
  directions: every falsy non-boolean (`None`, `0`, `""`, `[]`) selected
  best-effort recording without being a declared spelling of it - measured at 75
  of 100 attempted frames written, 25 counted in `dropped_frame_count` and
  `save_episode` completing - and every truthy non-boolean selected fail-fast and
  then reported `strict=True` in the refusal text whatever the caller wrote.
