### Removed: the unwired per-unit teleop input envelope leaves rather than reading as enforcement

The per-unit teleop input envelope this branch had added to
`mesh/security.py` (`INPUT_ENVELOPES_BY_UNIT`, `NORM_MODE_UNITS`,
`input_envelope_for_units`) is dropped: nothing called it, nothing tested
it, its `rad` row was byte-identical to the `deg` row it existed to differ
from, and the comment justifying it described `DEFAULT_INPUT_VALUE_ABS` as
a `4*pi` radian assumption when the constant has been `720.0` frame units
on this tree throughout. Unwired safety machinery reads as a solved problem
to the next reader, so it leaves with AGENTS.md convention 10. The
`value_abs_by_key` / `max_slew_by_key` parameters stay as the extension
seam and now say plainly that nothing populates them yet. Wiring it to the
receiving robot's declared `norm_mode` is tracked in #2935, where it can
carry its own mixed-unit tests instead of riding a dashboard change.
