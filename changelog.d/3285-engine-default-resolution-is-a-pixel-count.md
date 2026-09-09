### Fixed: the engine default render resolution is a pixel count at the owner that stores it

`MuJoCoSimEngine`, `NewtonSimEngine` and `IsaacSimulation` all take
`default_width` / `default_height`, and none of the three graded them - while
`add_camera` and the render family refuse the same values on every backend. The
constructor is not an inert default: MuJoCo copies it into the `SimCamera` entry
registered for the free camera and for every model camera a robot brings, so a
resolution `add_camera` refuses had a second, ungraded way into the same field.
Measured on a so101 scene, `default_width=0` published
`observation["default"]` as a `(480, 0, 3)` zero-pixel frame under
`status="success"`; `-1` and `inf` dropped the image key entirely, handing a
policy that declares `requires_images` proprioception alone; and `True`, `2.7`,
`640.0`, `'640'` and `nan` raised `TypeError` out of `get_observation`. Every
`render()` - a call passing no dimensions - was refused for a value its caller
never named. On Isaac the legacy spelling ran through `int(...)` into the graded
`camera_width` field, so the coercion defeated that domain: `True` stored a
1-pixel camera and `2.7` stored 2. All three constructors now grade both
dimensions on the shared `positive_count_error` floor and raise `ValueError`
naming the class, the parameter and the value. The framebuffer *ceiling* stays a
render-time check, because it is a property of the compiled model.
