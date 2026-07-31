### Fixed: `use_rosbridge` reports a port its WebSocket transport cannot address

The tool accepts the OS port range `1-65535`, but the client underneath gates
the port with `type(port) == int and port in range(0, 65535)`. Two values the
tool accepts failed that gate and left it as a bare `AssertionError`, out of a
function whose contract is a result dict: port `65535`, and an `int` subclass
such as an `IntEnum` at any value, including the default `9090`. Both escaped
every action and `RosbridgeRobot.drive` with them.

An `int` subclass is now normalized so it reaches the wire - it is a legal port
and dials as the equal plain `int` does - and the residual bound failure is
reported through the tool's error envelope, naming the transport and the
usable range. The accepted domain is unchanged, so it still agrees with
`RosbridgeRobot`, the inference server CLI and the mesh session checks.
