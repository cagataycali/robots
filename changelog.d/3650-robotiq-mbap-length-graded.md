### Fixed: a Robotiq reply is read by a length the codec has graded

`_ModbusTcpClient._exchange` sized its body read straight from the MBAP length
field the peer declared. Everything else about that header was graded -
`parse_response` checks the protocol id, the transaction id and the length
against the frame - but all of it runs *after* the read that field sizes, so a
peer that is not a Modbus server was reported as a gripper that stopped
answering. Pointed at an HTTP endpoint, the driver waited out its whole timeout
and said `activation failed: timed out`, while the codec's own
`protocol id must be 0, got 21584 - this is not Modbus TCP` never fired; a
declared length of 0 was reported as a truncated response rather than as the
illegal length it is.

Worse, the field bounded nothing: a socket timeout applies per `recv`, not to
the loop reading a declared 65535 bytes, so a peer dribbling bytes held the
connection - and the client lock with it - indefinitely. Measured against a
peer sending one byte every 0.4s, a driver configured with `timeout=2.0` was
still blocked when the measurement was abandoned at 45s.

New `mbap_body_size(header)` grades the header at the door that reads by it,
beside every other field's domain in `protocol.py`: the protocol id, and the
length against `MIN_MBAP_LENGTH..MAX_MBAP_LENGTH` (2..254, the unit id plus at
most a 253-byte PDU). All four cases above are now refused from the seven header
bytes alone, naming the field, in under a millisecond.
