### Fixed: an empty `cmd_scale` resolves from its owner table, not from a bare `1.0`

`WBCConfig.__post_init__` requires `cmd_scale` to carry exactly three entries
`[vx, vy, omega]` *when provided*, and admits an EMPTY sequence as "not provided"
(`if cmd_scale_length and cmd_scale_length != 3`). But this field's default is not
empty - it is the full upstream triple `_DEFAULT_CMD_SCALE = (2.0, 2.0, 0.5)`. So
omitting the argument and passing an empty sequence were two spellings of one
request that resolved to different scales, because both command-block builders
fell back to a bare unit scale for a vector too short to slice:
`scale = cmd_scale[:n_vel] if cmd_scale.shape[0] >= n_vel else np.ones(n_vel)`.

With `target_velocity=[0.5, -0.25, 2.0]`, omitting `cmd_scale` commanded
`[1.0, -0.5, 1.0]` while `cmd_scale=[]` commanded `[0.5, -0.25, 2.0]` - `vx`/`vy`
HALVED and `omega` DOUBLED - under a `success` result. A length-TWO `cmd_scale`
was refused by name the whole time (`must have exactly 3 entries`), which is what
made the empty one a hole rather than a policy: the one wrong length that was not
reported is the one that silently substituted a scale no table states. Pre-fix,
`cmd_scale=[]` and an explicitly stated `cmd_scale=[1, 1, 1]` produced an
identical block, so "not stated" could not be told from "unit scale".

That second fallback number is the failure the sibling scale field in the same
module already fixed: `obs_scales` is completed from `_DEFAULT_OBS_SCALES` at
construction, and `build_single_frame` resolves an omitted key from that same
table rather than a bare `1.0`, because a second fallback "would silently
multiply the 29 joint-velocity entries of the frame by 20 ... which the network
reads as a malformed observation". The velocity scale is the other half of that
idea, and the command block is the observation's FIRST `command_dim` entries, so
a dense network carries it to all `num_actions` joint targets - measured over one
`get_actions` tick, all 15 leg+waist targets differed between the two spellings
(max 0.0134 rad), and now none do.

`__post_init__` completes an empty `cmd_scale` from the upstream table, so
`config.cmd_scale` is always the vector the block is built with, and both
`_resolve_command` implementations resolve an unsupplied component from that
table through one inherited `_velocity_scale` - replacing two byte-identical
copies of the `np.ones` fallback, so the non-gait and gait blocks cannot be built
with different scales. A `cmd_scale` a caller does state is still honored,
including a deliberate unit scale, and every existing refusal (wrong non-empty
length, scalar, non-finite component) is unchanged.
