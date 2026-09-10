### Fixed: the Feetech bus reads the whole arm in the `SYNC_READ` frame it is named for

`FeetechBus.sync_read` sent one unicast `READ` per servo. `SYNC_READ` (0x82) is
the one Feetech instruction where a broadcast expects replies - one frame names
the register and the IDs, and every addressed servo answers with a status packet,
back to back - and the method's own docstring gave the reason for not using it as
"the SCS SYNC_READ instruction is not available on every servo in this family".
That is the wrong way round: the instruction is unavailable on the *SCS* series,
which is protocol 1 and which this codec does not address at all. lerobot keys
the same refusal on the same per-model protocol number, and every servo this bus
carries is protocol 0.

Two costs came with the per-servo read, measured through real `pyserial` against
a pty playing six STS3215s (host-side cost only - no baud limit is simulated):

| port | read window | before | after |
| --- | --- | --- | --- |
| plain | 1.00 s (the default) | 6067 ms/read, **0.16 Hz** | 10.1 ms/read, 98.6 Hz |
| plain | 0.05 s | 362 ms/read, 2.76 Hz | 10.1 ms/read, 98.7 Hz |
| echoes the host's frame | 0.05 s | **0-1 of 6 joints**, every read | 6 of 6 joints, 98.6 Hz |

The first two rows are the read asking the port for 10 bytes when a two-byte
register reply is 8: `serial.Serial.read(n)` returns early only once `n` bytes
arrive, so every servo waited out the entire read window for two bytes no servo
sends. The third row is the same over-request on a half-duplex adapter that
echoes: the 10 bytes are 8 of echo and 2 of the reply, the frame is truncated,
nothing verifies, and the arm publishes no joints at all - the failure
`bus_access.read_joints` documents from hardware, where an arm published zero
joints for eleven hours while its presence stayed healthy.

A six-servo read also paid the 10 ms reply settle once per servo, putting a 60 ms
floor under every state read - below the 30 Hz the joint consumers named in
`read_joints` publish at. It is now one settle for one frame.

The six joints are also now one pose: they come from replies to a single packet
rather than six round trips, so a moving arm no longer reports joint 1 and joint
6 as though they were sampled at the same instant when they were 50 ms apart.
`write_goal_positions` already gives this reason on the write side.

New in the codec, each graded byte-for-byte against `scservo_sdk`'s own
`GroupSyncRead`: `Instruction.SYNC_READ`, `sync_read_packet`,
`sync_read_reply_size` (the vendor's reply budget, `(6 + width) * motors`, which
is what the port is now asked for), `parse_sync_read_replies` (frames the reply
stream by cutting each packet at its own `LEN` and handing the slice to
`parse_status_packet`, so there is one framing implementation), and
`STATUS_OVERHEAD`. A servo that does not answer, answers a frame that fails its
checksum, or is not on the bus is absent from the reading rather than guessed at,
and a retry re-asks only the servos still missing.
