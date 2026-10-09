### Tests: the camera pixel-count sweep drives each surface with five probes, not all thirteen values

Every camera surface on MuJoCo, Newton and Isaac (constructor default,
`add_camera`, the render family, Newton's `open_viewer`) refuses a bad width
or height by calling `positive_count_error` itself. The full thirteen-value
table is now pinned once on that guard, and each surface gets one probe per
way it could drop the guard (`0`, `-4`, `2.7`, `True`, `"big"`). 480 cells
become 161 with the same package lines executed.
