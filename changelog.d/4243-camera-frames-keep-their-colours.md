### Fixed: a camera frame on the mesh had its red and blue swapped

The mesh camera publisher handed RGB frames (MuJoCo's renderer, lerobot's
cameras) to an encoder that reads BGR, so every dashboard card showed a red cube
as blue and a blue sky as brown. The publisher now converts before encoding;
raw and single-channel frames are untouched.
