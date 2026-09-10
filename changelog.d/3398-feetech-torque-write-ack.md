### Fixed: a Feetech torque write is confirmed by the servo, not by the absence of an error

`FeetechBus.set_torque` wrote `Torque_Enable` to each servo and reported a motor
failed only on `OSError` from the host's own port. The six-byte status packet a
unicast `WRITE` returns - the reply `write_packet` documents, and the one the
vendor SDK sets a packet timeout for after every write - was never read, so two
things followed.

Nothing measured the claim the driver makes about it. `FeetechDriver`'s
`set_torque` and `stop` verbs refuse with "these motors did not answer and may
still be driven", and that list could only ever be empty: an unplugged, mute or
garbling servo was reported as released, and a `stop` on an arm whose elbow had
lost its connector answered `torque_enabled: false` for a joint still holding
its position.

And the unread acks were left for the next reader. Six of them sit 36 bytes in
front of the following `SYNC_READ` reply stream; on a port that reports
`in_waiting` the read path's top-up recovers them, but on one that does not, the
stream is read short - a healthy six-servo arm answered a state read with one
joint and logged the other five as not replying.

The ack is now read back per motor and framed through
`parse_sync_read_replies`, which skips a half-duplex adapter's echo of the
host's own frame rather than refusing it as bytes in front of a status packet. A
motor that answered nothing or answered something that does not verify is named
in the return, so the refusal has a measurement behind it and the bus is left
clean for the next frame. The frame itself now comes from `write_packet` instead
of being built inline, so the reply-expecting write is spelled once.
