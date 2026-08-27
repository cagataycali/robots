### Fixed: the arms page's real-hardware and sim-asset claims are graded against the registry

`docs/robots/arms.md` closes with a *Compatibility notes* block whose bullets are set-membership
claims about `registry/robots.json`: which arms ship no simulation asset, and which drive real
hardware. A reader plans hardware work from them, so a stale name there is a wrong answer rather
than a cosmetic one. All three claims had gone stale.

The real-hardware bullet read "`panda`, `so100`, and `ur5e` are also supported on real hardware via
LeRobot. The rest are simulation-only". Only `so100` was right. Neither `panda` nor `ur5e` declares a
`hardware` block at all, so `Robot("panda", mode="real")` refuses with `Unsupported robot type:
'panda'`, and LeRobot registers no Franka type among its sixteen -- the page told a reader the exact
opposite of the truth about the arm family it names first. "The rest are simulation-only" then
misdescribed the nine arms that do have a path: `dynamixel_2r`, `hope_jr`, `koch`, `omx`, `openarm`,
`rebot_b601`, `so101`, `vx300s`, `wx250s`. The sim-asset exception named two of its three arms,
omitting `rebot_b601`, which declares no `asset` block either.

Nothing graded any of it. `tests/test_docs_robot_catalog_coverage.py` reads the same page and its
four guards pin catalog-table *membership*, the robot *counts* and the *Aliases* column; a capability
claim in the prose block is none of those, which its own docstring is precise about.

All three bullets are now derived from the repo's own sources of truth rather than restated, so an
arm that gains or loses a path fails the bullet that should have named it: LeRobot from the entry's
`hardware.lerobot_type`, and native from `drivers.registry.get_native_driver_class`. Because the
LeRobot list is read from the registry alone it grades on an install without LeRobot, and a
LeRobot-gated premise pins the assumption that makes that sound -- every type the arms declare is one
LeRobot registers. A second premise pins the coincidence the split currently rests on: every arm with
a `hardware` block names a type today, and `reachy_mini` shows a driver-only block is expressible, so
the day an arm does that the file says the two routes have come apart rather than letting the LeRobot
bullet gain a robot LeRobot cannot build.

Deliberately unchanged: the block's last bullet, about what the `joints` count includes. Of the 59
registry robots whose asset compiles, 50 declare a `joints` value other than the asset's actuator
count, so that field is a loose informational number whose contract needs deciding before it can be
graded -- a different question from which arms drive hardware.
