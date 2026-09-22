### Fixed: the SO-101 cuRobo demo offers only a URDF cuRobo can build a model from

`examples/so101_curobo` plans to a `gripper_frame_link` tool frame and measures
its grasp offsets in it. The SO-ARM100 revision the strands-robots asset cache
pins declares no such link, so `--planner curobo` with no flags handed cuRobo a
URDF its builder refuses (`Link gripper_frame_link not found in parent map`),
swallowed the crash in the scripted fallback, and loaded the sim arm from that
URDF too - leaving the scene with no cameras and the recorded dataset with no
image features. The same resolver passed cuRobo the cache's `assets/` subdir
while the URDF spells its meshes `assets/<f>.stl` relative to itself, so none
resolved.

A cached URDF that declares no tool frame is now declined with that reason and
the remedy, an explicit one is refused before cuRobo is imported, and the mesh
search path is the directory the URDF's own refs resolve under.
