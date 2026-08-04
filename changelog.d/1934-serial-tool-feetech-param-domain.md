### Fixed: serial_tool refuses a Feetech parameter its packet cannot carry

The three Feetech actions pack caller-supplied values into fixed-width fields of
a servo packet - `motor_id` becomes the address byte, and `position` / `velocity`
are written as `value & 0xFF` and `(value >> 8) & 0xFF` - and nothing bounded
them first. That packing *reduces* an out-of-range value rather than refusing it,
so a request the servo could not honor became a different command that it could,
reported as a success. Measured against the bytes reaching the bus:
`feetech_position(position=65536)` put goal **0** on the wire (full travel to one
end) while reporting `Position 65536 (5761.4 deg)`, `position=-1` put **65535**
there, `position=70000` put **4464**, and `-2048` put **63488**. `velocity` wraps
the same way: `-1` reached the bus as `65535` and `65536` as `0`. Because the
success text echoes the value that was *asked for* rather than the one that was
sent, the report could not be used to notice it.

`motor_id` was equally unbounded despite its documented 1-254 range: `0` and
`255` reached the bus as an address byte, `True` addressed **motor 1** while
reporting `Feetech Motor True`, and `256` / `-1` / `2.7` dead-ended in
`bytes must be in range(0, 256)` and `bad operand type for unary ~: 'float'` -
messages naming neither the tool nor the parameter that was wrong.

Each of the three is now checked against the range of the field it is written
into, before the port is opened, so a value that cannot be carried never
energizes a motor. The bounds are read off the protocol rather than chosen:
`motor_id` gets the 1-254 range the tool already documents, `position` gets
0-4095 - the scale this tool already reports on, since it echoes every write back
as `position / 4095 * 360` degrees - and `velocity`, which declares no travel
limit, is bounded by the two-byte width of the register it is packed into. Only
the parameters an action actually writes are checked, so a `read` or `send` is
never refused for a servo value it ignores, and a parameter that was not supplied
at all still gets its existing "required" message.

The range test itself is a new shared `strands_robots.utils.bounded_count_error`,
which `tcp_port_error` now binds instead of re-implementing: an upper bound is
load-bearing wherever a value is packed into a fixed-width field, and a second
copy of that rule would agree with the first until one of them was changed.
`bool` is rejected there for the reason a hand-rolled range test tends to miss
it - it is an `int` subclass, so `minimum <= value <= maximum` lets `True`
through as a silent `1`. Port refusals are unchanged, down to the message text.
