### Fixed: `pose_tool` frames a servo reply with the Feetech codec, not a copy of it

`pose_tool`'s `read_position` located and verified the `Present_Position`
status packet with its own parser beside
`strands_robots.drivers.feetech.protocol`. It now frames the reply with
`parse_sync_read_replies`, so the tool and the driver read one wire format
through one implementation. The one check the copy had and the codec lacked
moves into the codec: `parse_status_packet` refuses an error byte with bit 7
set (`ProtocolError`), which `scservo_sdk`'s `rxPacket` also treats as payload
rather than a reply.
