### Fixed: `DatasetRecorder.create` refuses a camera frame shape it cannot honor

`camera_dims` and the `video_width` / `video_height` pair are one quantity in two
spellings -- the recorder declares each camera at
`camera_dims.get(camera, (video_height, video_width))` -- and neither was
checked. The shape is a declaration rather than a resize, so whatever was given
went straight into the LeRobot feature as `(3, height, width)` and was not
compared against a real frame until the first `add_frame`.

The quiet one was an entry keyed by a name `camera_keys` does not declare: that
lookup never finds it, so the camera it was meant for silently took the global
pair instead. A camera streaming 240x320, declared as
`camera_dims={"imagee": (240, 320)}` against `camera_keys=["image"]`, was
declared `(3, 480, 640)` from the defaults -- nothing logged, the dataset
created. A component that is not a positive integer was written in as given, so
the schema declared `(3, 480, nan)`, `(3, 480, '640')` or `(3, 480, True)`; and a
value that is not a two-element sequence unpacked as a bare `TypeError`, a
non-mapping `camera_dims` as a bare `AttributeError` from the lookup, and a NumPy
integer width as `TypeError: Object of type int64 is not JSON serializable` from
the metadata write -- none of which named the parameter or the method.

Both spellings now go through the shared `positive_count_error` per component,
with `camera_dims` additionally held to being a mapping whose every key is a
declared camera and whose every value is a `(height, width)` pair. The strict-int
domain is the one the consumer honors: a pixel count is written into
`meta/info.json`, where an integral float lands as `480.0`. The refusal sits in
the same guard block as the schema column names, ahead of the same two side
effects -- the lazy lerobot import, so one caller mistake reports identically on
a minimal install, and the on-disk target, which `overwrite=True` removes.

`camera_dims=None` / `{}` still mean "not supplied", a list pair is still
accepted, and with no camera declared neither spelling is read, so nothing that
could be honored is refused. `fps` is the recorder's other unchecked schema
option; it is a rate rather than a frame shape and already has a named owner at
the `start_recording` facade, so it is tracked separately.
