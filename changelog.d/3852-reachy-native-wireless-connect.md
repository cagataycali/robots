### Fixed: a native Wireless Mini connects, and the driver registers as an agent tool

`ReachyDriver.connect_eagerly()` refused every Wireless Mini that was not handed
an explicit Zenoh bridge, naming the transport keyword as the remedy. Daemon
1.10.0 serves the same `/ws/sdk` endpoint on Wireless as on Lite, so the driver
now takes that socket when no transport is supplied; passing `transport=` still
selects the Zenoh bridge, and daemons without `/ws/sdk` still need it.

`ReachyDriver` carried the whole agent-tool surface - `tool_name`, `tool_spec`,
`stream` - without inheriting `AgentTool`, so `Agent(tools=[driver])` logged
`unrecognized tool specification` and registered nothing: the driver was silently
absent from `agent.tool_names` rather than refused. It now subclasses `AgentTool`.

Three wire-level corrections travel with it. Antennas are sent `[right, left]`,
the order the daemon's `SetAntennasCmd` documents, so a named `antenna_left` no
longer drove the opposite side. The daemon's seven head-motor readings are
separated into `body_yaw_deg` and six `head_leg_deg`, rather than all seven being
published as Stewart legs. `WebSocketLink.stop()` closes with `close_timeout=1.0`
and awaits its cancelled reader, so teardown finishes inside the driver's
five-second cleanup budget instead of the websockets ten-second default, and a
reader that failed no longer leaves the socket behind.

Hardware evidence is read-only: connection, joint and IMU telemetry, the emotion
catalogue, agent dispatch and a clean disconnect on daemon 1.10.0. Motion is not
verified on hardware. Native camera, audio and pixel-directed look continue to
refuse by name.
