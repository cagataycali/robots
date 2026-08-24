### Fixed: `serial_tool` bounds a servo register write to the field it encodes into

The Feetech register writes masked the caller's value into fixed-width packet
bytes (`value & 0xFF`, `(value >> 8) & 0xFF`), so an out-of-range value was not
refused - it was silently truncated into a different, reachable command while the
success message quoted the value the caller supplied. `position=70000` put 4464 on
the wire and `position=-1` put 65535, the largest the two-byte field holds, both
reported as `success`. `motor_id=255` built a frame whose ID byte duplicates the
header, `motor_id=True` silently addressed motor 1, and `motor_id=300` leaked
`bytes must be in range(0, 256)` from inside the packet builder.

The same call also forwarded `baudrate`, `read_bytes` and `timeout` unchecked:
pyserial coerces `baudrate=2.7` to 2 baud, returns no bytes for a non-positive
`read_bytes` (indistinguishable from a timed-out read), waits no time at all for
`timeout=nan`, and overflows its deadline for `timeout=inf`.

Every numeric option an action consumes is now checked before the port is opened,
so a refused call reaches no bus at all, and only the options that action reads
are checked. `timeout=0` stays valid as pyserial's documented non-blocking poll.
