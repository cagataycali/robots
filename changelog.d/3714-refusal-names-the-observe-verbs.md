### Fixed: the real robot tool's unknown-action refusal names the verbs that read the arm

The `action` enum and the unknown-action refusal were two literals over one
vocabulary. The observe verbs were added to the enum while the refusal went on
naming only the motion ones, so `action="get_stat"` came back with
`Valid actions: execute, start, status, stop` - a list from which every verb
that reads the arm was absent. The next thing an agent could reasonably do to
read a joint angle was request a policy rollout on real actuators, which is the
harm the observe actions exist to remove. Both readings now come from one
`_PUBLISHED_ACTIONS` tuple, so a verb cannot be published without being named.
