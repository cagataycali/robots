### Fixed

- **A fresh training start clears an empty `output_dir`, not a directory that holds a checkpoint.**
  `LerobotTrainer.train()` cleared a stale `output_dir` on a fresh (`resume=False`)
  start whenever no *resumable checkpoint* was visible, using a recursive
  `shutil.rmtree(..., ignore_errors=True)` that reports neither what it took nor a
  partial failure. The `lerobot_train` tool implements the same hygiene bounded to
  an EMPTY directory, and "no resumable checkpoint" is strictly weaker than
  "empty": lerobot's `save_checkpoint` writes `model.safetensors` before
  `train_config.json`, so a run interrupted between the two leaves the trained
  weights under a checkpoint no resume probe reports - and `output_dir` is a
  caller-supplied path on both entry points, including the `train_policy` agent
  tool. Measured on a real ACT checkpoint, a fresh start removed 1,239,111,004
  bytes of trained weights and then refused the call for an unrelated reason,
  having trained nothing. Both entry points now ask one owner,
  `strands_robots.utils.stale_output_dir_is_clearable()`; a directory holding
  anything is left for lerobot to refuse by name. Emptiness subsumes the
  checkpoint probe, so the tool drops a redundant test and keeps its previous
  verdict on every input.
