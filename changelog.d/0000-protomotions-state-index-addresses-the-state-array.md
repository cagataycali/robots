### Fixed: ProtoMotions reads a flat `observation.state` through the index `set_robot_state_keys` builds

`set_robot_state_keys` built a joint-name -> `observation.state` offset map and
cached it on the policy, and its docstring said so. Nothing read it. The
convention that needs it rescanned the key list with `list.index` for every one
of the 29 joints instead - twice a tick, positions then velocities, for the whole
rollout. The map is now what the read goes through, and it is rebuilt with the
list rather than beside it, because a map that outlived its list would address a
re-ordered feed at the old offsets: a fully-populated but permuted pose, which is
the one error resolving by name exists to prevent. First occurrence wins, as
`list.index` answered, so a feed that repeats a key reads the same entry as
before.

The refusal that fires when the array has no offset for a key named a private
attribute and prescribed a call the caller had already made:

    KeyError: ProtoMotionsPolicy: joint 'left_hip_pitch_joint.vel' missing from
    self._robot_state_keys (call set_robot_state_keys).

A joint-name key list addresses positions. `set_robot_state_keys` validates that
every joint the config names is present, so after a successful call a position is
always addressable and only a suffixed spelling - a velocity - can be missing;
repeating the call with the same joint list adds nothing. The refusal now names
the spelling it could not address, states that the array is addressed by the list
that method was given and how many keys that list holds, and names the three
routes that do supply velocities: `dof_vel=` through `get_actions`, per-joint
`<joint>.vel` keys on the observation, or widening the state array and its key
list together. Each is driven in
`tests/policies/protomotions/test_observation_state_reads_the_same_through_every_convention.py`,
so the message cannot name a route that does not work.
`docs/policies/protomotions.md` documents the flat-state route beside the
per-joint one.
