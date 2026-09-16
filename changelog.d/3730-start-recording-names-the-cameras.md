### `start_recording` names the cameras the dataset records

The reply said `6 joints, 1 cameras @ 10fps` - a count, not a name - and an
agent that needed the name next asked `render` for `top_camera` and was told the
only camera was `default`. The schema line now names the cameras on every
backend, by their SCENE name (`2 cameras ['default', 'so101/wrist'] @ 10fps`) -
the spelling `render`, `get_frame` and `cameras=` accept - and states the
dataset column beside it when `camera_schema_key` collapsed a `/` to `__`, so
neither name has to be guessed. When no camera is recorded, a second line says
what the dataset will carry (joint state and actions, no
`observation.images.*`) and reads the reason off the scene: a camera-less scene
is pointed at `add_camera(...)`, a `cameras=` that scoped every camera out is
told which ones it scoped out, and a scene whose cameras produce no frame is
sent to neither.
