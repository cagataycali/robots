### Fixed: `add_robot` names a robot that starts inside the ground plane

On a flat world, 13 of the 145 simulatable registry robots spawn with geoms
inside the ground plane (LeKiwi 34.6 mm, Unitree A1 120 mm, UR10e 27 mm) and
the contact solver pushes them out on the first step, behind a plain `success`.
`add_robot` now measures the burial after spawning and adds a `Warning:` line
with the depth and the `position=` that spawns the robot resting on the ground,
and logs the same line, so `Robot(name)` names it too. The spawn itself is
unchanged.
