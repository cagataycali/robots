### Added: `set_obs_noise` is reachable from the simulation tool schema

The MuJoCo dispatch router resolves an action with a bare `getattr`, so
`set_obs_noise` was dispatchable from Python while being absent from the
`action` enum in `tool_spec.json`, and none of its three noise magnitudes was
declared as a schema property. A model driving the tool reads that schema and
nothing else, so a capability that is implemented, documented, validated and
tested could not be configured by an agent at all.

The action is now advertised beside its sibling `randomize`, and
`joint_pos_std`, `joint_vel_std` and `camera_jitter_px` are declared as `number`
properties - `number` rather than `integer` because each is a float standard
deviation, and an `integer` declaration would tell a model that `0.01` is
invalid. `seed` was already declared, shared with `randomize`.

No library behaviour changes: the method, its validation and its noise path are
untouched. A regression pin derives the required property set from
`set_obs_noise`'s own signature, so a magnitude added later cannot become
undiscoverable by being omitted from the schema.
