### Changed: one lerobot registry walk, shared from `core`

`lerobot.<family>.__init__` deliberately imports none of its device
subpackages, so a draccus `ChoiceRegistry` has no choices until something walks
the family. Three modules did that walk with three copies of the same 50 lines -
`lerobot.robots` and `lerobot.cameras` in the hardware `Robot`,
`lerobot.teleoperators` in the `Teleoperator` factory, which also carried its
own copy of the third-party plugin loader the other two shared.

`strands_robots.utils.ensure_lerobot_family_registered(family)` is now the one
walk, beside `lerobot_version` where optional-dependency resolution lives.

The duplication was load-bearing for the layer graph. `Teleoperator` reaches for
the robot registry on one refusal path - to tell a caller that `so101_follower`
is a follower, not a leader - and reached it by importing a private helper of the
hardware `Robot`. That single edge was what kept `teleoperator` in the `app`
layer, which in turn made the teleoperation mixin's read of the factory an
upward import: `drivers|mesh -> app`, deferred into `attach_teleop` to keep it
legal. With the walk shared from `core`, the factory sits beside the mixin that
attaches what it builds, and `app` has no reader below it left:

```
upward deferred edges: 5 -> 4      (drivers|mesh -> app group: 1 -> 0)
```

`tests/test_import_layers_are_a_dag.py` now grades edges into `app` of every
kind, not runtime only - a deferral is still a dependency - so a module moved
back up fails the pin rather than being written into the roster.
