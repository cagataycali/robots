### Fixed: `add_camera` holds a camera's name to the alphabet its consumers can carry

A sim camera's `name` was checked only for being an addressable registry key, so
a name none of the consumers that key its frames can carry registered under
`status="success"`: on one `create_world`, `add_camera` accepted `'..'`,
`'sub/../etc'`, `'a b'`, `'cam#1'`, `'*'`, `'**'`, `'a//b'`, `'/lead'` and
`'trail/'`. Those names reach the mesh frame topic
`strands/<peer_id>/camera/<name>`, where `*` and `**` are Zenoh wildcards a
`put` is routed by intersection, and the S3 object key, where `..` walks out of
the peer's own prefix. All three simulation backends now read one rule
(`scoped_camera_name_error`), which is the bare-token alphabet the hardware
`cameras` doors already require plus one optional robot scope: the multi-robot
form `add_camera("alice/wrist_cam", ...)` is unchanged, because the mesh strips
exactly that one namespace level before publishing.
