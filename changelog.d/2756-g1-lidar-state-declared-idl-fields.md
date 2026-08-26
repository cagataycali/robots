### Fixed: the G1 LiDAR state record reads the fields `LidarState_` declares

`_on_lidar_state` reached for `msg.code` and `msg.freq`. `LidarState_` declares
neither. The MID-360 reports its fault code as `error_state` and its scan rate
as `cloud_frequency`, so the two readings the record exists to carry were being
taken from names the message never had.

Nothing about that failed. Each read went through `getattr(msg, name, default)`,
and a name the IDL does not declare simply yields the default - a well-formed
value that lands in the published record indistinguishable from a measurement.
So `strands/<peer>/lidar/state` carried `code = -1`, `code_text =
"-1 (unknown)"` and `freq = 0.0` on every tick, and a unit whose LiDAR had
genuinely faulted at 10 Hz published exactly the same record as a healthy one.
Of the record's four readings only `sys_rotation_speed` was a declared field
and therefore real. Fed a real `LidarState_` carrying `error_state = 3` and
`cloud_frequency = 10.0`, the mesh now publishes `code = 3` and `freq = 10.0`
where it previously published `-1` and `0.0`.

`error_state` is read once and feeds both the numeric code and its rendered
text, so the two cannot drift into describing different fields. The text still
renders through `decode_code`; that table holds SDK RPC and loco/arm response
codes rather than LiDAR faults, and every entry in it other than success is
numbered above the widest `uint8`, so for a field declared `uint8` it can only
ever report success for zero or fall back to the bare integer - it cannot
invent a meaning for a LiDAR fault.

The existing test could not have caught this. Its stand-in was built by hand to
spell the names the decoder happened to read, so it agreed with the decoder
whatever those names were; it now spells the declared names. Field-name
fidelity is graded separately against a frozen copy of the declaration, which
is itself checked against the real `LidarState_` wherever `unitree_sdk2py` is
importable - that SDK installs from a git clone rather than PyPI, so it cannot
be a test dependency and the frozen copy is what lets the rule hold in CI.
