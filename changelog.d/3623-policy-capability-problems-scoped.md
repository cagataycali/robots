### Fixed: a policy type that does not resolve gets no claim about its capabilities

`LerobotTrainer.validate()` refuses an `extra['policy_type']` lerobot does not
register, and then went on to grade the spec's capability knobs against it.
Each capability is probed off the policy's own lerobot config class and falls
back to a static set when the type is not in the registry, and a misspelled
name is in no such set - so `relative_actions`, `method='expert_only'`,
`embodiment` and `tune` components each reported "not supported by policy_type
'<typo>'" and named the types that DO expose the field, describing the
configuration of a policy the same response said does not exist. The remedy
carried the harm: each problem prescribes dropping the knob, and a typo of a
policy that supports it (`pi5` for `pi05`, `grot` for `groot`, `smolvl` for
`smolvla`) led a caller to delete a legitimate setting while fixing the name.
The capability checks are now scoped to a policy type that resolved, the way
the reward-model gate scopes its field check; checks that grade the request
itself - the method spelling, the `tune` key spelling and value domains,
`sample_weighting` fields - stay unconditional.
