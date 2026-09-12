### Fixed: a camera's name is held to one alphabet at every door that accepts one

`Robot(cameras={...})` validated every option of a camera entry and never the
name it was keyed under, while `lerobot_teleoperate` refused a `robot_cameras`
name that is not a bare token. The name is what every consumer keys the camera's
frames by, so the ten names measured as accepted by the factory each land as
structure somewhere downstream: `Mesh` publishes frames on
`strands/<peer_id>/camera/<name>`, where `/` adds a topic level and `*` is a
Zenoh wildcard a `put` is routed by; `CameraOffloader.s3_key_for` joins the name
into the object key, where `..` walks out of the peer's prefix; and a recording
writes it as the `observation.images.<name>` dataset feature key. A camera named
`wrist/ref` published its 274054-byte inline frame on
`strands/rover-01/camera/wrist/ref` - the shape of the small S3 pointer the IoT
transport exempts from its camera-frame drop - so the whole frame reached the
broker that drop exists to spare.

The rule now has one owner, `utils.camera_token_error`, which both doors read:
letters, digits, `_` and `-`, opening on a letter or a digit. A name every
consumer can carry is unaffected.
