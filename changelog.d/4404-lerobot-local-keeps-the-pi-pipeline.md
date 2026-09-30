### Fixed: `lerobot_local` keeps a pi0 / pi0.5 checkpoint's own processor pipeline instead of degrading it

Four faults on the same path, measured with `lerobot/pi05_base` and `pi05_droid`.
`dim_policy="pad"` padded the state but not the action, so every 32-D pi checkpoint
(`max_action_dim` padding; the arm's joints are the first N) was refused by every
embodiment - "6 action_keys but model action dim is 32" - and `pad` / `truncate` now
take the first N values of a wider action (`strict` stays exact). A declared embodiment
that could not be configured discarded the whole pipeline with a warning, so the run
continued without normalization and pi0.5's `Task: ..., State: <tokens>;` prompt became
the bare instruction; it is now refused at load with `ValueError`, naming the cause
and `camera_key_map=` / `obs_rename_override=` / `set_robot_state_keys`. The raw
path refused a partial camera set for the types that are built for one (pi0, pi0-FAST,
pi0.5, SmolVLA, X-VLA: `accepts_partial_images`); it now lets them mask the missing
views and still refuses no view at all. And a pipeline whose `TokenizerProcessorStep`
tokenizes the instruction now makes the policy report that it reads it, so
`run_policy` no longer tells an agent a pi0.5 "does not read the instruction". The
discard-only helpers (`_adopt_pipeline_tokenizer`, `_embodiment_config_failed`,
`state_key_remedy(embodiment_rejected=)`) and their tests are removed.
