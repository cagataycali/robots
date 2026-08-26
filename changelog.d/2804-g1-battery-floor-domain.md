### Fixed: the G1's battery floor is held to a numeric domain before it is stored

`G1Driver.__init__` took `battery_floor_pct` and stored `float(battery_floor_pct)`
with no domain, and that floor is the only constructor value
`_check_motion_gates` compares a live reading against. `float("nan")` survives
the coercion, and `battery_pct < nan` is False for every reading, so the driver
stored a floor it reported through `get_status` and enforced nowhere: the gate
opened on a critically low pack. The string `"nan"` - how a config file spells
it - took the same path, and `True` became a silent `1.0%` because `bool` is an
`int` subclass. Measured on a connected driver whose FSM and lowstate were both
healthy, with the pack at 3.0%:

```
battery_floor_pct        construction   outcome at battery = 3.0%
15.0 (the default)       accepted       battery 3.0% is under floor 15.0%
float("nan")             accepted       gate open - the write was allowed
"nan"                    accepted       gate open - the write was allowed
True                     accepted       gate open - the write was allowed   (floor 1.0)
```

The two sibling numbers this driver takes, `duration` and `n_steps` on
`run_policy`, already go through a shared domain from `strands_robots.utils`.
The floor now goes through `finite_number_error`, the shared domain whose own
contract covers this hazard, before the coercion rather than after it - a guard
placed after `float()` would judge a value already stored, and would turn the
named refusals for `None`, a list and a value past the float64 range into bare
`TypeError`/`OverflowError` escaping the constructor's documented contract.

Every value that behaves as `battery_pct < floor` reads is untouched: `0.0` and
`-10.0` still never trip, `500.0` still always does, and a NumPy float still
passes. Whether a percentage should be bounded to `[0, 100]` is a separate
question about what a percentage may mean rather than a correction, and it is
deliberately left open.
