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
reference configuration, its velocities and actuator setpoints to zero, and
nothing else in the world moves. `reset()`, whose contract *is* a world-wide
reset, is unchanged.
