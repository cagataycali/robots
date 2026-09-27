### Fixed: under `STRANDS_MESH_BACKEND=bridge`, e-stop and resume now reach the MQTT/IoT leg

The bridge's Zenoh leg opens the process-wide Zenoh session, so the safety path
found a session zid and published `strands/safety/estop` and
`strands/safety/resume` on a raw Zenoh publisher. That publish never entered
`BridgeTransport.put()`, so the cloud/IoT side never saw a stop although both
topics are in the default bridge filter. Under the `iot` and `bridge` backends
the envelope now goes through the transport's `put()` (body-level HMAC binding),
reaching every leg the transport serves.
