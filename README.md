# Cosmos 3 diffusers checkpoint-mismatch artifact

Measured on Thor (NVIDIA Thor, torch 2.11.0+cu130, MUJOCO_GL=egl) with
`nvidia/Cosmos3-Edge`.

* `measured_prefix_diffusers_0.39.0.json` - on `main`, diffusers 0.39.0: the load
  raises a bare `NotImplementedError: Cannot copy out of meta tensor` naming
  neither the library, its version, nor the checkpoint.
* `measured_postfix_diffusers_0.39.0.json` - with this change, same diffusers:
  a `RuntimeError` naming the checkpoint, the installed diffusers, the 112
  unfilled tensors and the upgrade command.
* `measured_postfix_diffusers_0.40.0.dev0.json` - with this change on a diffusers
  that supports the checkpoint: loads (0 of 745 parameters unfilled) and runs
  forward dynamics in 2.23 s at 8.98 GB peak.
* `capture.py` / `compose.py` - the capture and the figure generator. `compose.py`
  asserts every rendered number against the JSON dumps, that the generated video
  is not static, and that the figure border is clean.
