### Fixed

- `docs/training/vla_workflow.md`: the GR00T example put `embodiment` inside `extra=`, which `Gr00tTrainer` never reads (it needs the `TrainSpec.embodiment` field to emit `--embodiment_tag`), so the copy-pasted block returned `status="error"`. The example now sets the field; a test executes the documented block and runs it through `Gr00tTrainer.validate`.
