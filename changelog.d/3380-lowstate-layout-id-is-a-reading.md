### Fixed: the G1 lowstate layout id is a reading, not an `int()` cast

`mode_machine` arrives on `rt/lowstate` beside the IMU and is echoed on every
`LowCmd_` the G1 driver builds. The firmware drops a frame whose layout id does
not match the one it announced, so an id built from something other than a
number is a write the robot silently ignores - the same failure a typed default
has elsewhere in this decoder, on the one field that leaves the process again.

The four IMU vectors in that method read through
`strands_robots.drivers.base.telemetry_int` / `telemetry_float_list`; the layout
id beside them was still coerced with a bare `int()`. Measured through
`_on_lowstate`:

| `LowState_.mode_machine` holds | before | after |
| --- | --- | --- |
| `9` | `9` | `9` |
| `True` | **`1`** | `None`, previous kept |
| `False` | **`0`** | `None`, previous kept |
| `b"\x09"` | raises into the shared `except` | `None`, previous kept |
| `"n/a"` | raises into the shared `except` | `None`, previous kept |
| `2.7` | `2` | `2` |
| `"9"` | `9` | `9` |

`1` and `0` are valid uint8 layout ids, so a flag on the field was
indistinguishable from a reading. `telemetry_int` refuses both. A float is still
truncated: `telemetry_int(2.7) == 2` is the owner's own pinned answer and the Go2
accepts it too, so refusing it here would be the drift those functions exist to
prevent.

A refused reading leaves `_mode_machine` at the last value that parsed rather
than clearing it, matching `_refresh_fsm_id` two ranges over. Today the raise
position produced that too; stating it keeps it true once a field is decoded
after this one, which is what went wrong on the IMU side.

`tests/drivers/test_g1_layout_id_is_a_reading_too.py` pins the refusals, the
accepted boundary including the float rows, and an AST scan for any non-`None`
`getattr` default across the whole method on both Unitree drivers - a derivation
rather than a search for this defect's own constants, since the constants a
future field would default to are not knowable.

Pure routing: `g1.py`'s executable statement count is unchanged at 558.
