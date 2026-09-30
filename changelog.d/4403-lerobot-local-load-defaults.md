### Fixed: `lerobot_local` runs on the GPU and leaves a checkpoint's `torch.compile` off unless asked

Two fields of a checkpoint's `config.json` decided how it ran at inference.
`device` records where the checkpoint was trained or saved - `lerobot/pi05_droid`
ships `"cpu"` - and `lerobot_local` honoured it when no `device=` was given, running a
4-billion-parameter pi0.5 at 7-10 s per chunk on an idle GPU. It now picks the best
device present (CUDA, then MPS, then CPU) and warns when that differs from the
checkpoint's field; `device=` still wins. `compile_model: true` (the LIBERO pi0, pi0.5
and pi0-FAST fine-tunes, `max-autotune`) wrapped the model in `torch.compile`, so the
first inference spent 8+ minutes autotuning inside the control loop with nothing
logged; it is now off for inference with a warning, and the new `compile_model=True`
keeps it (saying the first inference takes minutes). Both are applied to the config
before the policy is built, and the model cache keys on `compile_model`.
