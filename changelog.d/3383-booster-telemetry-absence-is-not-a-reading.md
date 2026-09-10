### Fixed: a Booster T1 field the frame does not carry is not a zero reading

`parse_low_state` documents the rule - "absent fields are omitted rather than
defaulted: a snapshot that reports a zeroed IMU the robot never sent is worse
than one that reports none". #3381 applied it to the three IMU vectors; the four
motor vectors were still `float(getattr(motor, <field>, 0.0))` and `_on_battery`
read its three fields the same way, so the one function held two conventions.
Measured through the decoders, all 42 cells of remaining field x non-reading
reported a value the frame did not carry:

| the frame | before | after |
| --- | --- | --- |
| `q` renamed or dropped | all 23 joints at `0.0` | `joints=None` |
| `q` arrives as a flag | all 23 joints at `1.0` | `joints=None` |
| `soc` renamed | `battery_pct=0.0` - an empty pack | `battery_pct=None` |
| `soc` arrives as a flag | `battery_pct=1.0` - a one-percent pack | `battery_pct=None` |

The `battery_pct` rows are what `get_status` already reads for: it publishes
`battery.get("pct")`, so it was written to report `None` for a T1 that reported
no charge, and the writer made that unreachable.

The `joints` row moves the robot rather than only misreporting it. It is the
hold source - `send_action` reads it as `held_q` and `build_frame` writes
`held_q[slot]` as the position target of every *uncommanded* upper-body joint. A
full-width vector of defaulted zeros is finite and non-empty, so it passed both
of `_on_low_state`'s guards, was cached, and the next write commanded all eight
arm joints to exactly zero. Measured end to end on a frame whose `q` was
renamed: before, `send_action` returned `status=success` and published one
`LowCmd` collapsing the arms out of the pose the robot was holding; after, it
returns the refusal the driver already spells for this case ("no LowState frame
has arrived yet, so neither the frame width nor the hold position of an
uncommanded arm joint is known") and publishes nothing.

Coercion was also per *message* rather than per field, so one unreadable value
discarded every field beside it. A `q` that raised in `float()` cost the
velocities, torques and temperatures in the same frame, leaving `_last_state` on
the previous one - a staleness that reads as a dropped wire rather than as one
renamed field.

The motor and battery reads now go through the owner of this rule, as the IMU
reads already did. `strands_robots.drivers.base.telemetry_float` and
`telemetry_float_list` decide what counts as a reading for the Unitree drivers,
and the T1 reads every field through the same two functions - so a value one
driver refuses cannot be a reading on another.
The record keeps the shape those drivers use - the key stays and the value is
`None` - so a consumer asking for a field always gets an answer and the answer
can be "the robot did not report this". The three field maps are named
(`MOTOR_STATE_FIELDS`, `IMU_STATE_FIELDS`, `BATTERY_STATE_FIELDS`) so the read is
derived once rather than restated per field, and the vectors stay all-or-nothing
because `held_q` is indexed by slot: a vector short one element would renumber
every slot after the gap and hold the wrong joint at each of them.

`tests/drivers/test_booster_telemetry_absence_is_not_a_reading.py` pins the rule
as a table derived from those maps, the frame surviving one bad field, the hold
source end to end, and a derivation that refuses a typed default returning to the
module.
