### Fixed: a servo velocity that would reverse the command is refused

`Goal_Velocity` (register 0x2E) is sign-magnitude on the Feetech STS/SMS series:
bit 15 carries the direction and bits 0-14 the magnitude. `serial_tool` bounded
the field to the two-byte maximum instead, keyed to the byte width rather than to
the register, so every magnitude from 32768 up was accepted and put its own two
bytes on the wire - where the servo read them as a command in the opposite
direction. `velocity=65535` ran the joint at full speed the wrong way,
`velocity=40000` ran it in reverse at magnitude 7232, and `velocity=32768` read as
magnitude zero and stopped a servo the caller had just asked to run. Each returned
`status="success"` quoting the number supplied, so the report described a command
the servo never received.

This is the same silent reinterpretation the sibling fields are bounded to
prevent, and it is not a truncation: the caller's value reached the wire intact
and simply meant something else there. The field is now bounded to the largest
magnitude that leaves the direction bit clear, derived from that bit rather than
restating a number, and the refusal quotes the reason so a caller can see why
32767 is the ceiling.

`Goal_Position` (0x2A) shares the encoding and needed no change: its 12-bit
ceiling of 4095 already sits far below the direction bit, which is what a bound
keyed to the register rather than to the byte width buys. `docs/hardware/tools.md`
documented the old `[0, 65535]` domain and is corrected with it.
