### Fixed: the mesh pages call `subscribe(topic, callback, name=)` in the right order, and a name-first call is refused

`docs/learn/mesh/topics.md` and `fleet.md` showed
`a.mesh.subscribe("imu", "strands/arm-b/imu", lambda key, payload: ...)` - the name
first - while the signature is `subscribe(topic, callback=None, name=None)`. Run as
written it subscribed to the literal key `imu`, so the callback never fired, stored
the lambda as the subscription name, and the next `mesh.stop()` raised
`TypeError: sequence item 0: expected str instance, function found`. The examples now
pass the topic first and the name by keyword, and `subscribe` refuses a non-string
or empty topic, a non-callable callback ("did you pass the name first?") and a
non-string name with `TypeError`, before anything is recorded.
