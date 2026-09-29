### Fixed: a description short name from `list_discoverable()` loads the curated robot

`list_discoverable()` names each MJCF robot by its `robot_descriptions` module
(`so_arm101`, `viper`, `widow`, `n1`, `openarm_v1`). Five curated entries ship
one of those modules without claiming its short name, so `Robot("so_arm101")`
fell through to discovery and loaded the same MJCF without the curated metadata.
The joint labels were empty, the `hardware` block was gone, and the labelled
`send_action({"shoulder_pan": ...})` that the first-robot guide teaches was
dropped. `so101`, `vx300s`, `wx250s`, `fourier_n1` and `openarm` now list those
names as aliases, and one registry test requires every curated entry's
description short name to resolve to that entry.
