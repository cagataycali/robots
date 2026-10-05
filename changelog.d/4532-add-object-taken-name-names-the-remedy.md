### Fixed: a taken `add_object` name is refused the same way on every backend, with the remedy

Adding an object under a name already in the scene now answers `add_object:
object 'cube' already exists. Remove it first (remove_object).` on MuJoCo,
Newton and Isaac, like `add_camera` does. MuJoCo used to say only
`Object 'cube' exists.`.
