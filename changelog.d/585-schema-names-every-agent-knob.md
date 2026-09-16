### Fixed: the simulation tool schema names every knob an agent can pass

`randomize`'s `color_range`/`friction_range`/`mass_range`, `start_recording`'s
`overwrite`, `replay_episode`'s `action_key_map` and
`start_cameras_recording`'s `max_frames_per_camera` were accepted at runtime
but missing from the schema, so an agent could not plan with them; they are now
typed and described, and the `randomize` flags say what they do.
