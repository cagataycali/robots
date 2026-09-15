### `start_recording` names the cameras the dataset records

The reply said `6 joints, 1 cameras @ 10fps` - a count, not a name - and an
agent that needed the name next asked `render` for `top_camera` and was told
the only camera was `default`. The schema line now lists the dataset's camera
keys (`1 camera ['default'] @ 10fps`) on every backend, and when there are none
a second line says what the dataset will carry (joint state and actions, no
`observation.images.*`) and why - `cameras=[]` scoped them all out, or the
scene has no camera to `add_camera(...)` first.
