### Fixed: a Feetech servo's reply is located and verified, instead of being read at fixed offsets

`pose_tool`'s `read_motor_position` read the two position bytes from indices 5 and 6 of whatever
the bus returned. The servo bus is half-duplex and shared by every motor on the arm, so a reply
can arrive behind a byte the host's own transmission echoed, or be the late answer to a read that
already timed out, sent by a different motor. Both shift those indices, and the result is a joint
angle that is wrong rather than absent: a joint at 1024 counts (-89.98 degrees) was reported as
-180.0 when one byte preceded its reply, and the tool quoted that as a measurement. A reply from
another motor, a corrupted checksum and a frame one byte short were all accepted as well.

The reply is now found and verified the way `scservo_sdk` -- the vendor SDK that owns this wire
format, and which the `[lerobot]` extra already installs -- does it: the `FF FF` header is
searched for rather than assumed at offset 0, the frame length is re-derived from `LEN`, the
checksum is recomputed over the same span the outgoing packet signs, and the responding ID must be
the one that was asked. Leading bytes are recovered from rather than refused, because bytes in
front of the header do not make a reply corrupt, they make it offset -- the SDK reads the true
1024 from exactly those bytes, and so does this now. A frame that cannot be verified reports no
position, which is the signal every caller already handles: the `read_position` action answers
with an error envelope instead of a number, and the interpolating paths decline to build a
trajectory. Graded cell for cell against the SDK across nine framings, the two now agree
everywhere.

This was a motion fault and not only a reporting one. `_smooth_move` reads the current position
before dividing the travel into increments, so a shifted reply made the arm interpolate from a
place it was not: commanding a joint at 1024 counts toward 4095 wrote `[0, 511, 1023, 1535]`,
opening a *smooth* move with a ninety-degree lurch away from the target. It now writes
`[1024, 1279, 1535, 1791]`, byte for byte the trajectory a clean bus produces. `incremental_move`
applied its delta to the same fiction and now refuses instead.

Acting on a servo's error byte -- declining to drive an overheating motor -- is deliberately not
decided here. A value inside the byte's range is a valid frame, as the SDK also holds, so a
reported fault still reports its position; bounding the *value* to the register's 12-bit range is
likewise a separate question from framing, and the suite pins that this parse answers neither.

Nothing could have caught this: three test fixtures across three files built the reply with `0`
where the checksum belongs and a hardcoded ID of 1, a frame no servo would send and one the vendor
SDK calls corrupt. Each now carries a real checksum and answers as the motor the outgoing read
addressed, so a misattributed position is visible to a test for the first time.
