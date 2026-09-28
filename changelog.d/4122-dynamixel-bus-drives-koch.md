### Added: a native Dynamixel bus, so `Robot("koch", mode="real", driver="strands")` reads, moves and runs a policy

`DynamixelBus` puts the Protocol 2.0 codec on a serial port: one `SYNC_READ`
of four-byte `Present_Position` for the whole arm, one `SYNC_WRITE` of
`Goal_Position`, and an acknowledged `Torque_Enable` write per motor, with
frames identical to `dynamixel_sdk`'s. `DynamixelDriver` is `FeetechDriver`
over that bus - the same verbs, degrees and percent-open units,
`lerobot-calibrate` records and 30 Hz `run_policy` - and is registered for
`koch`. aloha, vx300s, wx250s, trossen_wxai and dynamixel_2r stay refused until
each has a verified motor map. The codec gains `write_packet`,
`sync_read_packet` and `parse_status_stream`.
