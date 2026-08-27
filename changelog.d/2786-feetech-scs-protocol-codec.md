### Feature: native Feetech STS/SCS Protocol 1 codec for the STS3215 family

`strands_robots.drivers.feetech.protocol` frames and parses the wire format the
STS3215 and its SCS/SMS siblings share, without opening a serial port. Every
builder (`ping_packet`, `read_packet`, `write_packet`, `sync_write_packet`)
returns exactly the bytes the vendor SDK (`scservo_sdk` on PyPI) would put on
the wire, and `parse_status_packet` accepts what the SDK's `PacketHandler`
accepts. The additive checksum (`~sum(payload) & 0xFF`) is the identity
`strands_robots.tools.pose_tool` already uses inline; stating it as a function
lets the parser grade the same bit pattern it built.

Two audiences share the suite. 45 of its 47 cells build the datasheet-published
byte sequences by hand and drive the codec directly, so every framing, domain
and status-parsing shape is graded whether or not `scservo_sdk` is present. The
other 2 are `TestTheVendorAgreesOnFraming`, which runs only where the SDK is
installed and compares the codec's output byte-for-byte against the SDK's own
`PacketHandler` internals.

The bus and driver skeleton land in follow-up PRs (issue #360 scopes 2 and 3);
this PR ships only the codec, in the same slice the harness triage names as
"landable-without-hardware" and the same slice #2750 shipped for Dynamixel.

Refuses at the boundary:

- IDs outside `0x00..0xFE`, non-integer IDs (including `bool`, since `bool` is
  an `int` subclass and a motor addressed as `True` is a caller bug the wire
  cannot tell from `1`)
- Broadcast writes without an explicit opt-in, so a caller who addresses a
  broadcast where a reply is expected is refused here rather than by a servo
  that never answers
- Sync-writes with duplicate motor IDs or mismatched per-motor block sizes
- Status packets that are truncated, have trailing bytes, mismatch the
  addressed ID, mismatch the expected LEN, or fail the additive checksum

Every refusal is documented and catchable. All six exported callables
(`build_packet`, `ping_packet`, `read_packet`, `write_packet`,
`sync_write_packet`, `parse_status_packet`) can hand a caller a `TypeError` or a
`ValueError`, so each names both in its own `Raises:` block - including the ones
raised a frame down in `_validate_id`, which is where a caller who passes
`motor_id="3"` to `build_packet` actually receives its `TypeError`. `ProtocolError`
is exported alongside them, because every parser docstring names it as the class
that separates a wire fault from a caller bug and a handler the package will not
hand out is a contract the caller cannot write. It subclasses `ValueError`, so a
bus that wants wire faults alone writes `except ProtocolError` and one that wants
every refusal writes `except ValueError`.

Importing the package does not pull `scservo_sdk` (grader:
`tests/drivers/test_feetech_module_load.py`), matching the module-load pin the
Dynamixel driver package carries against `dynamixel_sdk`.
