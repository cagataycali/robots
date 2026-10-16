### Fixed: a Feetech bus read costs the reply, not the read timeout

`FeetechBus` asked pyserial for two bytes more than a status frame holds, so every `ping`, `read`
and `sync_read` waited out the 50 ms serial timeout: a real SO-101 observed at 18 Hz at most. The
bus now reads the 4-byte head and then exactly `LEN` more bytes, skips an echoed request and a stray
byte ahead of the header, and its timeout is 20 ms. `Driver.read()` on six servos drops from 50 ms
to under a millisecond.
