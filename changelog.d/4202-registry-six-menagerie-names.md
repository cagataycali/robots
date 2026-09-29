### Added: `mujoco_humanoid` in the registry; `gen3`, `iiwa14`, `iiwa`, `tiago` and `tiago_pp` resolve by name

`Robot("gen3")` and `Robot("iiwa14")` - the names `robot_descriptions` ships
those Menagerie models under - did not match a registry entry or alias, so they
fell through to discovery as category `discovered` with a synthesized
description while `kinova_gen3` and `kuka_iiwa` sat in the registry with the
same asset. Both are now aliases (with `iiwa`), and `tiago_dual` gains `tiago`
and `tiago_pp`, the regex-safe spellings of its `tiago++` module. MuJoCo's own
reference humanoid (`google-deepmind/mujoco` `model/humanoid`, pinned through
`mujoco_humanoid_mj_description`) is a first-class `humanoid` entry,
`mujoco_humanoid`, with aliases `humanoid` and `mjc_humanoid`, a robot page, a
nav row and a streamed 3D view. `kuka_iiwa` declares 7 joints: the 11 it
carried matched nothing in its compiled model.
