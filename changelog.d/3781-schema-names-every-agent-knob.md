### Fixed: the simulation tool schema names every knob an agent can pass

Thirteen parameters were accepted at runtime and missing from the schema, so an
agent could not plan with them: `randomize`'s `color_range`/`friction_range`/
`mass_range`, `start_recording`'s `overwrite`, `replay_episode`'s
`action_key_map`, `start_cameras_recording`'s `max_frames_per_camera`, and the
rollout knobs `control_substeps`, `reset_between`, `async_rtc`,
`rtc_inference_timeout_s`, `wbc_install_torque_control`,
`max_onframe_failures` and `policy_kwargs` (the goal payload cuRobo / MoveIt2 /
WBC read `target_pose` from). All are now typed and described, and the
`randomize` flags say what they do.
