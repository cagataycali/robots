### Fixed: `add_robot` no longer rewinds the scene it is added to

`add_robot` gave the robot it had just merged in a defined starting
configuration by calling `mj_resetData` over the whole world, so it also reset
everything else: an arm parked with `send_action` lost its pose *and* the
actuator setpoints holding it (it then collapsed under gravity over the
following steps), an object that had settled or been carried somewhere teleported
back to its declared spawn, a latched `apply_force` wrench was dropped, and the
simulation clock rewound to zero - all reported as a successful add, and all in
direct contradiction of this method's own documented contract ("preserves
previously-created world state"). The world-wide reset is also what forced the
home-pose re-apply that used to follow it, a partial repair that only covered
robots spawned with a `keyframe=` and overwrote wherever such a robot had since
been driven.

The reset is now scoped to the robot being added: its joints go to the model's
reference configuration, its velocities to zero, and nothing else in the world
moves. `reset()`, whose contract *is* a world-wide reset, is unchanged.

Removing the world-wide reset also exposed something it had been hiding.
`spec.recompile` transfers simulation state POSITIONALLY, and while it defines
the new `qpos` (from `qpos0`), `qvel` and `act` entries, it leaves the new `ctrl`
entries uninitialized - so any recompile that adds actuators produced setpoints
whose value was whatever the fresh allocation happened to contain (observed
across runs as denormals through `4.6e+228`). That is not a harmless nonsense
number: MuJoCo stops actuating the *entire* model on any step where a single
`ctrl` value is non-finite, so one uninitialized entry could silently release
every held pose in the scene, on the runs where the leftover memory happened to
be NaN. Those entries are now defined as zero as part of the recompile, before
anything reads them, which also removes the intermittent
`Nan, Inf or huge value in CTRL` instability warning that scene mutations could
already emit. Those entries are defined positionally - every index the transfer
left untouched - rather than per robot, because the two answer different
questions: a robot's actuator ids say *which robot may command an actuator*,
while the tail says *which entries were never written*, and MuJoCo's check reads
the whole buffer. Ownership is also not guaranteed to cover the tail: it is empty
for an actuator that is neither namespace-prefixed nor joint-driven, such as the
fixed tendon that couples a gripper's fingers.
