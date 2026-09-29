### Added: `flux3_action` - Black Forest Labs FLUX 3 Action as a policy provider

`create_policy("f3a")` (alias `flux3_action`) resolves to a provider around the
`flux-action` package and the published SO-101 checkpoint. The provider owns the
unit frame the model was trained in and the one the simulator and LeRobot
drivers speak: degrees to radians, gripper percent to the jaw angle, and the two
sign and offset conventions that differ (`lift = 90 - model`, `elbow = model +
90`), with a round-trip test pinning each. Actions come out as a 30 Hz chunk
queued behind one `select_action`, so a ~5 s `flex-fna` replan only stalls the
queue rather than the control loop.

Construction is cheap: the 22 GB checkpoint is not touched until `load()`, the
first `reset()` or the first `get_actions()`, in the same shape as
`lerobot_local`, so a registry sweep that instantiates every provider by name
does not download or load weights. The `flux3` extra carries the install hint
the refusal names when `flux-action` is missing; on Jetson Thor a NATTEN shim
falls back to dense attention where the wheel has no kernel.
