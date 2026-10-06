### Fixed: `start_task` refuses an ONNX policy provider with no `onnx_path` before it connects the arm

`microduck` and `protomotions` refuse to build without an ONNX weight, but the
registry listed nothing under their `requires`, so the pre-build guard in
`start_task` and the driver rollouts let them through: the arm was connected
and the build then failed on the worker thread. Both now declare `onnx_path`
as required, like `rsl_rl_onnx`, and the refusal says what to pass: a local
`.onnx` path, or for microduck a shipped weight name like `alpha_walking.onnx`.
