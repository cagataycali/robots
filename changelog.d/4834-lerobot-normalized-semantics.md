### Fixed: a LeRobot-normalised deployment (`so101-normalized-v1`) attaches to an SO-101

A LeRobot policy server whose descriptor says `semantics: "so101-normalized-v1"` (the label servers write for a
checkpoint trained on LeRobot-normalised data) was refused at attach with `unknown LeRobot semantics`, and the remote
path has no `units=` to work around it. `units/semantics.json` now reads `so101-normalized-v1` and
`so100-normalized-v1` as `same_as` the existing range/gripper-percent rows (arm `range_m100_100`, gripper
`range_0_100`). An alias is read, never written: strands still labels its own deployments
`so101-range-gripper-percent-v1`.
