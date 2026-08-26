### Fixed: the G1 low-level write path applies the vendor's per-joint gains, not one scalar

`G1Driver.send_action` builds a `LowCmd_` for `rt/lowcmd`, filling `kp`/`kd` for
any joint the caller did not give explicit gains for. Those defaults were a
single pair -- `kp=25.0`, `kd=0.5` -- applied to all 29 joints. The G1's
low-level position mode is tuned against a per-joint gain table whose `Kp` takes
three distinct values and `Kd` two, because the joints do not carry comparable
loads: both knees are the stiffest entries at `kp=100, kd=2`, and they are the
joints that hold a standing biped up.

Every entry of the scalar pair sat strictly below every entry of the reference
table, so no joint was even coincidentally correct. The deficit arrived in three
distinct ratio pairs -- 1.6x stiffness and 2.0x damping on the ankles, waist and
arms; 2.4x and 2.0x on the hips and waist yaw; 4.0x and 4.0x on both knees --
which is why no single scalar, chosen anywhere in the table's range, can
reproduce it.

Nothing surfaced this. The firmware validates `crc`, `mode_machine` and the
per-motor Enable byte; it does not validate gains. An under-gained frame is
therefore well formed, accepted, and reported as a successful publish, so the
only symptom is a robot that tracks its target badly or sags under its own
weight, with no error for an operator to read. Gains also sit inside the
checksum payload -- the vendor's `LowCmd_` pack format spends `B3x5fI` per motor
on mode, padding, `q`/`dq`/`tau`/`kp`/`kd`, and reserve -- so they have to be
correct before the CRC is stamped rather than corrected in a later frame.

The gain tables are now grouped the way the vendor groups its own lists (two
legs, a waist, two arms), so the 29-entry width follows from 6 + 6 + 3 + 7 + 7
and the left/right symmetry is visible rather than asserted. Behaviour a caller
already relied on is unchanged: a supplied `kp`/`kd` still wins, including when
only one of the two is supplied, in which case the omitted term falls back to
that joint's own reference value rather than to a flat default. The zero-torque
stop frame still zeroes every gain, which is what "soft" means on that wire.

This also strengthens the test that claimed to pin the defaults. Its docstring
said the wire capture made a change to them "a test-visible event, not a silent
regression", while the assertion imported the two constants it was comparing
against -- an expectation read from the constant under test follows any edit to
that constant, so it stayed green for every value including a wrong one. The
expected numbers are now literals, and the new regression file grades the whole
table against a locally stated copy of the reference, slot by slot, without
needing the vendor SDK installed.
