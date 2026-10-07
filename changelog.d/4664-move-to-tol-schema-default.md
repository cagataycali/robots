### Fixed: the sim tool schema states `move_to`'s real `tol` default, 0.015 m

The `tol` parameter description in the MuJoCo tool schema said `move_to`
defaults to 0.01 m, while both the MuJoCo and Isaac `move_to` use 0.015 m. An
agent planning a retry from the schema now reads the number the backend
converges on, and a test ties every default the description states to the
method signature it describes.
