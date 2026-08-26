### Added

- **drivers/dynamixel**: native Protocol 2.0 codec and a driver skeleton
  satisfying `HardwareDriver` for Koch, ALOHA, ViperX 300s (`vx300s`),
  WidowX 250s (`wx250s`), Trossen WidowX AI (`trossen_wxai`) and the
  Dynamixel 2R educational arm. `Robot("koch", mode="real", driver="strands")`
  now resolves natively instead of raising the "no native driver registered"
  error the driver seam produces for a robot without one. Motion, task and
  policy paths deliberately do not land yet; each returns a named
  "not wired yet (issue #359 bus)" envelope of the same shape a successful
  call would return, so a caller writes the same error-handling code either
  way. The bus that would carry the writes (serial I/O, sync-read timing,
  hot-reconnect) is scope item 1 in issue #359 and is its own PR - the same
  slice issue #354's triage recommends and issue #360's triage explicitly
  names as landable-without-hardware. The codec (`build_packet`,
  `sync_write_packet`, `parse_status_packet`, `checksum`,
  `decode_model_number`) is pure and verifiable against `dynamixel_sdk`
  byte-for-byte; the WRITE-LED, PING and SYNC_WRITE frames the tests pin
  reproduce Robotis SDK output exactly (CRC 0xE6CC, 0x4E19 and the full
  0xEF40-terminated sync-write frame respectively). `decode_model_number`
  returns the number register 0 reports and nothing else: mapping that number
  to a model name is per-model hardware metadata, gradable only against a live
  servo, so it lands with the bus rather than as a table this codec's tests
  could check for shape but never for truth.
- **drivers/dynamixel**: the codec escapes the reserved `FF FF FD` byte run.
  Protocol 2.0 forbids that run inside a payload because a servo reads it as
  the start of the next packet; the protocol escapes it with an extra `0xFD`,
  counts the inserted byte in `LEN`, and takes the CRC over the stuffed frame.
  All three codec entry points now do this and `parse_status_packet` reverses
  it, so the "byte-for-byte against `dynamixel_sdk`" property this module
  claims holds for every payload rather than only for payloads that happen not
  to contain the run. The gap was reachable: `GOAL_POSITION` is a signed
  32-bit little-endian value and -131073 -- exactly -32.0 turns at 4096 counts
  per revolution, an ordinary multi-turn target -- encodes to `FF FF FD FF`.
  It is the only value in the legal extended-position range
  (-1048575..1048575) that does, which is what makes an unescaped write a
  once-in-two-million data-dependent fault rather than something a first
  bring-up would catch.
