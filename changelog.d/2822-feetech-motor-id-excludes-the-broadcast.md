### Fixed: `serial_tool` no longer accepts the Feetech bus broadcast as a single servo's ID

`motor_id` was bounded at 254 and gave the byte width as the reason ("the packet carries the
ID in one byte, and 255 is the frame header"). Two of the 256 values that byte holds address
no servo at all: `0xFF` is the frame header, and `0xFE` is the bus broadcast, which every
servo on the bus reads. Only the header was excluded.

All three of the tool's motor-addressed actions write to one servo and expect its reply, so
`motor_id=254` was not a wider version of the request but a different one. `feetech_position`
put a broadcast frame on the wire, driving every joint of the arm to the same angle while
reporting `Feetech Motor 254 -> Position ...` as though one had moved; `feetech_velocity` did
the same with a speed; and `feetech_ping` made every servo answer at once, colliding on the
half-duplex bus, then read the collision back as one reply. Each returned `status="success"`.

The vendor draws the line in the same place. `scservo_sdk` declares `BROADCAST_ID = 0xFE` and
returns `COMM_NOT_AVAILABLE` for `scs_id >= BROADCAST_ID` in each of the three operations
these actions perform - a ping, a single register read and a single register write - reserving
the broadcast for `SYNC_READ` / `SYNC_WRITE`, which this tool has no action for.
`strands_robots.drivers.feetech` states the same ceiling as `MAX_UNICAST_ID` beside its own
`BROADCAST_ID`; the value is restated in the tool rather than imported because importing it
executes `drivers/__init__`, which registers every driver and pulls in numpy, and a test pins
the two together so they cannot drift.

The ceiling is now 253, the largest ID that names one servo. The 253 values below it are
unchanged, 255 is still refused for the reason it always was, and broadcasting is unchanged:
`send` writes the bytes it is given and claims to address nothing, so a caller who wants a
broadcast frame still has one. The tool's documented range is corrected to `[1, 253]`.

`tests/tools/test_serial_tool_numeric_domain.py` parametrized 254 as an addressable ID, in a
class named `TestARegisterFieldIsNeverSilentlyTruncated`; that case now uses 253.
