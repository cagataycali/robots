### Fixed: the speed a serial bus is opened at is graded wherever a caller states one

`serial_tool` holds its `baudrate` to `positive_count_error` and records why: the
option is "coerced rather than checked by pyserial (`2.7` becomes 2 baud)". The
four constructors that reach the same `serial.Serial` held it to nothing -
`FeetechDriver` and `DynamixelDriver` converted it with `int()`, `FeetechBus` and
`pose_tool`'s motor controller stored it raw - so a speed that is not a count was
applied rather than reported, and the tool refused values the driver beside it
accepted.

Measured on a real pty: `baud_rate=0` opened the port **successfully** at a speed
no servo answers, so every read timed out indistinguishably from an unplugged arm;
`True` opened it at 1 baud and `2.7` at 2, while `get_status` reported the
converted number as the configured one; `-9600` / `None` / `[1_000_000]` reached
pyserial's own `Not a valid baudrate`, and `nan` / `inf` an `int()` conversion
error - each raised at connect time by a third library, naming neither the driver
nor the parameter.

All four now refuse a speed that is not a positive integer, naming the surface and
the option, and the conversions are gone: a value the domain admits is already an
`int`. `1_000_000.0` and `'1000000'` stop being accepted, which is the divergence
this removes - the tool surface already refused both.
