### Fixed: the Unitree drivers agree on which telemetry fields are readings

A DDS decoder reads its message with `getattr(msg, <field>, None)` and coerces
the result, so the coercion is what decides whether a field reaches the mesh as
a number or as `None`. Both decoders' docstrings state why that matters: a
*typed* default "looks like a reading", so a name the IDL never declared would
publish a plausible constant - a zero-amp pack, a zero-cycle count - for as long
as the robot runs.

That rule was written twice, as four functions with the same names in
`strands_robots.drivers.g1` and `strands_robots.drivers.go2`, and the two copies
did not agree on what a reading is:

| value on the field | `g1` before | `go2` before |
| --- | --- | --- |
| `True` on a scalar field | `1.0` / `1` | `None` |
| `bytearray(b"\x01\x02")` on a vector field | `None` | `[1.0, 2.0]` |
| `memoryview(b"\x01\x02")` on a vector field | `[1.0, 2.0]` | `[1.0, 2.0]` |
| `[True, 2]` on a vector field | `[1.0, 2.0]` | `None` |

Each copy guarded a class of value the other let through, and the third row is
the one neither guarded. `memoryview` is bytes-like and iterates as integers, so
a raw buffer landing on a field declared as a numeric vector decoded into a
two-element "quaternion" on both drivers - where four elements are declared, and
where the values are byte contents rather than a rotation. Measured through the
decoders: `_on_lowstate` cached `quaternion=[1.0, 2.0]` from
`bytearray(b"\x01\x02")`, `_on_sportmode` cached `foot_force=[1, 2, 3, 4]` from a
`memoryview`, and `_on_bms` cached `pct=1.0` from `soc=True` - a one-percent
pack, published on the mesh health wire, from a field that carried no reading at
all. Neither `go2`'s bytes-like guard naming two of the three buffer types nor
`g1`'s docstring claim that it "turns a bytes-like or string value into `None`"
was true of the values above.

The refusal branches of all eight functions were uncovered in both modules,
which is how two copies of one rule came to disagree unnoticed.

The rule now has one owner. `strands_robots.drivers.base` gains
`telemetry_float`, `telemetry_int`, `telemetry_float_list` and
`telemetry_int_list`, holding the union of both copies' guards plus `memoryview`,
and both drivers read through them - so a value one driver refuses cannot be a
reading on the other. The vector readers share one all-or-nothing walk, because
half a quaternion is worse than none: a consumer cannot tell that it is half.
Net -36 executable statements across the three modules (`base.py` +30, `g1.py`
-28, `go2.py` -38).

`tests/drivers/test_telemetry_coercion_refuses_the_same_non_readings.py` pins the
rule as a table, both decoders end to end, and a derivation over the driver
package that refuses a third private copy.
