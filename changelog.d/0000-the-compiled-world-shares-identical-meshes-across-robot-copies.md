### Fixed: robots of one model share one set of meshes in the compiled world

The MuJoCo backend calls `MjSpec.dedup_assets()` (MuJoCo 3.17+) before compiling, so N copies of a robot
reference one mesh, texture and material each instead of N. Eight so101 in one world now take 729 MB resident
(1185 MB before), about 17 MB per added arm (80 MB before); the compiled model stays at 62 MB instead of
growing by 62 MB per arm.
