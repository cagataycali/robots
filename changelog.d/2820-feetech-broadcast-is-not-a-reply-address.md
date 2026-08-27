### Fixed: a broadcast is refused where the tool is about to read one servo's reply

The Feetech Protocol 1 ID byte carries every unicast address and one address that
is no servo: `0xfe` is the broadcast. `serial_tool` bounded `motor_id` at 254 for
a reason keyed to the field's width - "the packet carries the ID in one byte, and
255 is the frame header" - which is true of the byte and says nothing about which
of its values a servo can hold. So `action="feetech_ping"` accepted the
broadcast: the tool wrote `FF FF FE 02 01 FE`, read ten bytes back and reported
`Feetech Motor 254 responded: <hex>`. Every servo on the bus receives that frame
and, because `PING` is answered, every one of them replies at once; on a
half-duplex bus those replies collide, so the bytes read back belong to no single
servo and the ID quoted beside them belongs to none either. With a single servo
attached it would appear to work, which is worse - it reads like a discovery
feature.

The address space is not this module's to decide.
`strands_robots.drivers.feetech.protocol` declares `BROADCAST_ID` and
`MAX_UNICAST_ID` ("Highest ID a specific servo may hold. 0xFE is the broadcast")
and already refuses the same intent from `build_packet` unless a caller passes
`allow_broadcast=True` "for `SYNC_WRITE` and other reply-less instructions that
legitimately target the broadcast ID". The tool now reads those two constants
rather than restating them, and applies that same rule at its own boundary,
before the port is opened.

A reply-less write keeps the whole range. `feetech_position` and
`feetech_velocity` never read a status packet, so addressing every servo with one
frame means exactly what it says and is still accepted; only an action that reads
a reply is held to a single servo. That set is derived from the tool's own body -
an action both reading the port back and consuming `motor_id` - so an action
added later that reads a reply for a caller-supplied ID is held to the same rule
the hour it lands. The refusal names the broadcast, why no single reply can come
back, and the range that can answer.
