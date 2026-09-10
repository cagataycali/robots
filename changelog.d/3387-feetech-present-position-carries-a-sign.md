### Fixed: `Present_Position` carries a sign, and the Feetech bus reads it as one

The STS/SMS series encodes `Present_Position` (0x38) as sign-magnitude: bit 15 is
the direction, bits 0-14 the magnitude. `READABLE_REGISTERS` spelled the sign per
entry - bit 15 for `Present_Velocity`, bit 10 for `Present_Load`, and `None` for
the register between them - so a servo reporting a joint just past its homing
zero had that direction bit read as the top of the magnitude. That is a routine
reading on a calibrated arm, and the degrees that came out were reported as a
measurement with no error. Measured through `FeetechBus.sync_read`, against
replies framed the way a servo frames them:

| joint | wire word | before | after |
| --- | --- | --- | --- |
| `shoulder_pan` | `0x0800` | 0.04 deg | 0.04 deg |
| `shoulder_lift` | `0x8064` | **1354.75 deg** | -94.40 deg |
| `elbow_flex` | `0x8320` | **2309.19 deg** | -208.61 deg |
| `wrist_flex` | `0x0064` | -85.60 deg | -85.60 deg |
| `wrist_roll` | `0x8001` | **2700.79 deg** | -180.09 deg |
| `gripper` | `0x0800` | 50.01 percent | 50.01 percent |

`shoulder_lift` spans -90..90 degrees; 1354.75 is not a value that joint can
hold. `Present_Load` on the same path was already correct, which is what made the
gap easy to miss.

The sign is now a table in the codec rather than a column of the reader's own:
`drivers.feetech.protocol.SIGN_BIT` maps each register to the bit that carries
direction, entry for entry with lerobot's `STS_SMS_SERIES_ENCODINGS_TABLE` - the
table an SO-arm is calibrated and read by - and `decode_sign_magnitude` applies
it. `READABLE_REGISTERS` now carries registers alone and looks the sign up, so a
register added there cannot pick "unsigned" by saying nothing.
`tools/serial_tool.py` reads its `Goal_Velocity` write ceiling from the same
table instead of restating bit 15, leaving one definition site for the
convention.

`max_magnitude` refuses a sign bit that is not a bit of the word: `0` leaves no
magnitude, and a bit at or above the word width masks nothing, so either one
would return the whole unsigned value and call it a magnitude - the reading being
replaced.

A reading below the joint's declared range is reported rather than refused or
clamped: a target outside the range is a caller's mistake, but a position outside
it is where the arm actually is.

`tests/drivers/test_feetech_position_carries_a_sign.py` pins the wire words
against lerobot's own decoder, grades the request and reply frames byte-for-byte
against the vendor `scservo_sdk`, and derives the "signed exactly where lerobot
says signed" check over every readable register rather than listing today's
three.
