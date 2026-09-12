### Fixed: a scene camera's name is held to the bare-token rule in every segment

`add_camera` on the MuJoCo, Newton and Isaac backends refused only a name that
cannot be a registry key at all, so `'..'`, `'sub/../etc'`, `'a b'`, `'cam#1'`
and `'**'` each registered under `status="success"` and reached the consumers
the hardware door was hardened against: the mesh published a frame on
`strands/rover01/camera/**` - a Zenoh wildcard on a `put`, routed by
intersection to every peer subscribed to any camera - and
`CameraOffloader.s3_key_for` joined `'../../etc/passwd'` into
`'frames/rover01/../../etc/passwd/123.jpg'`.

The hardware rule is not applied verbatim, because the simulation itself writes
a `/` into a camera's name: a robot's cameras register under
`<namespace>/<cam>` (`arm0/wrist`), `render_all` resolves a short name against
that form, and the dataset column collapses the separator to `__`. The scene
rule, `utils.camera_segments_error`, holds every `/`-separated segment to the
one alphabet both doors read - letters, digits, `_` and `-`, opening on a letter
or a digit - and refuses an empty segment. Every documented name, namespaced or
not, is unaffected; all three backends read the rule through
`camera_name_error`, so one refusal sentence covers them.
