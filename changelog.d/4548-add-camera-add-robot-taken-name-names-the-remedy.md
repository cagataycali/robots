### Fixed: a taken `add_camera` / `add_robot` name names the remover, like `add_object`

Adding a camera or robot under a name already in the scene now answers with the
verb and the call that frees the name on MuJoCo, Newton and Isaac:
`add_camera: camera 'wrist' already exists. Remove it first (remove_camera).` and
`add_robot: robot 'arm' already exists. Pick a different name, or remove it first
(remove_robot).` (MuJoCo also lists the existing robots). Isaac used to say only
`Camera 'wrist' already exists.`; Newton and Isaac said `Robot 'arm' already exists.`.
