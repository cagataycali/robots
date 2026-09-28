### Fixed: the hardware tool's `execute` and `start` accept `policy_config`

`Robot("so101", mode="real")` could name a policy provider but not configure
it: the tool handler dropped `policy_config`, so an agent could not point
`lerobot_local` or `flux3_action` at a checkpoint on hardware while the MuJoCo
tool already accepted the same field. The handler now validates
`policy_config` as a mapping and forwards it through the existing
`**policy_kwargs` seam to `_create_policy`, so the sim and real invocations of
one tool take the same arguments.
