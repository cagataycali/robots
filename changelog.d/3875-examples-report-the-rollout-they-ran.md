### Fixed: a demo never announces a rollout the runtime refused

`run_policy` reports a refusal by RETURNING `{"status": "error", ...}` rather
than raising. `examples/kimodo/kimodo_g1_walking.py` discarded that result and
printed `Done. Video: <path>` with exit 0 for a rollout that applied no action
and wrote no MP4 - which is what its own documented command does every time,
since the default `nvidia/Kimodo-G1-RP-v1` publishes bare weights and is
refused on the first sample; `examples/wbc/wbc_g1_gait.py` printed `gait
rollout complete` for any refusal. Both now read the status and raise, like the
`_check` helper `examples/09_procedural_terrain.py` already uses.
`docs/policies/kimodo.md` is corrected where it disagreed with the tree: the
Quick start needs `[sim-mujoco]` beside `[kimodo]`, the default `model_id` is
refused on the first sample rather than at construction, and the standalone
route is not the faithful visualisation of a sampled motion - the 29 joint
targets reach position actuators under gravity with the clip's root values
dropped, so a G1 handed a walking clip topples (pelvis 0.793 m to 0.064 m in
4 s at 50 Hz) where a `set_joint_positions` replay of the same clip holds it.
