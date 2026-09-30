### Fixed: an operator on the pure `iot` backend receives its direct replies; no Device Shadow mirror for operator and gateway peers

`init_mesh(..., peer_type="operator")` and the dashboard's gateway meshes
wired the presence shadow mirror like a robot, publishing
`$aws/things/<thing>/shadow/name/presence/update` on every heartbeat. The
shipped `strands-operator` policy grants no MQTT publish on `$aws/things/...`
(its shadow statement is the REST `GetThingShadow`/`UpdateThingShadow` pair),
and AWS IoT answers an ungranted publish by ending the session, so the
operator's `response/#` subscription died about once a second and every direct
reply with it: the documented operator path timed out on 5/5 `status` commands
while the transport blamed a missing child grant. `shadow.enable_for_mesh`
now skips peers whose `peer_type` is in `NON_ROBOT_PEER_TYPES` (`operator`,
`gateway`) with one INFO line; measured after the change, first direct reply
450 ms and 226 ms steady, zero session drops.
