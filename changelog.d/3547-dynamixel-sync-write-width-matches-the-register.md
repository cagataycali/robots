### Fixed: a sync-write's data_length is held to the register it addresses

`sync_write_packet`'s own documentation stated the requirement and named the
consequence of breaking it - `data_length` "must match the register's width in
`CONTROL_TABLE`", because "a 4-byte write to a 2-byte register is accepted by
the servo but writes into the next register" - but nothing enforced it. The
function checked that `data_length` was a positive 16-bit count and that every
entry's data was exactly `data_length` bytes, and never consulted the table it
cited.

So a 4-byte write to `GOAL_CURRENT`, a 2-byte register at address 102, framed:

    sync_write_packet(102, 4, [(1, (500).to_bytes(4, "little"))])
    -> fffffd00fe0c00836600040001f40100002059

`GOAL_VELOCITY` is the 4-byte register at address 104. The servo carves the
payload by address, so the top two bytes of that current command land in it: a
caller asking for 500 mA also commanded a velocity of zero on a joint that was
moving. The opposite mistake is as quiet - a 2-byte write to the 4-byte
`GOAL_POSITION` leaves half the target unwritten, so the joint goes to a
position nobody asked for. A SYNC_WRITE is a broadcast that no servo answers,
which is what makes both silent: neither can come back as an error, only as a
joint in the wrong place.

The width is now checked before the packet is framed, and the refusal says where
the bytes would have gone:

    GOAL_CURRENT at register_address=102 is 2 bytes wide, got data_length=4;
    the servo accepts the longer write and runs the extra 2 byte(s) on into
    GOAL_VELOCITY

Only an address `CONTROL_TABLE` names is graded. The table is a curated subset of
the servo's registers rather than an allowlist, so a write to a register it does
not list - `PROFILE_VELOCITY` at 112, say - carries whatever `data_length` the
caller asks for, and every register it does list still frames at its own
declared width. The widths are read from `CONTROL_TABLE` itself rather than
restated, so the check cannot drift from the table it grades against.
