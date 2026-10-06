### Changed: `Robot(name)` spawns a model authored below the ground resting on it

The thirteen registry robots that carry a `spawn_position` (LeKiwi, Unitree
A1/Go1/Go2, Aliengo, ANYmal C, UR10e, Open Duck Mini, Asimov v0, RB-Y1, and the
Aero, Allegro and Shadow hands) now spawn there when `position` is omitted, so
`Robot("lekiwi")` no longer starts 34.6 mm inside the ground and asks for the
number the registry already holds. Their docs pages drop the `position=`
argument. A `urdf_path=` or `keyframe=` spawn, and an explicit `position=`,
are unchanged; pass `position=[0.0, 0.0, 0.0]` for the old origin spawn.
