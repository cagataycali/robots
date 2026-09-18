### Fixed: a servo's `Present_Position` carries a sign on the pose tool's read

`Present_Position` (0x38) is sign-magnitude on the STS/SMS series: bit 15 is the
direction, which is what `drivers.feetech.protocol.SIGN_BIT` declares - entry for
entry lerobot's `STS_SMS_SERIES_ENCODINGS_TABLE` - and what the native
`FeetechDriver` bus reads. `pose_tool` decoded the whole two-byte field as a
magnitude, so a servo reporting a joint just past its homing zero, a routine
reading on a calibrated arm, was quoted as a position more than a full turn away:
`0x8064` is -100 counts, which `read_position` reported as 2709.49 degrees on a
`shoulder_pan` that spans -180..180. Nothing on a read path bounds the number it
quotes, so the value arrived as a measurement.

The frame was the other half of it. `pose_tool` and `serial_tool` each carried a
`build_feetech_packet` of their own, assembling `FF FF ID LEN INST <params> CHK`
and spelling the register address as a hex literal at every call site, so the
package held three copies of one wire format and the sign was nobody's to look
up. Both tools now frame through `read_packet` / `write_packet` / `ping_packet`
and name registers from `Register`. Every frame they put on the wire is
byte-identical to the one sent before, which is what leaves the sign as the only
behaviour that moved.
