### Fixed: a `MotionBricksConfig` synthesis knob the generator cannot integrate is refused

`MotionBricksConfig.__post_init__` guarded `fps` and `generate_dt` with a
comparison, and a comparison is not a domain: `nan < 1` and `nan <= 0` are both
`False`, `inf < 1` is `False`, and `True` is an `int` subclass satisfying `>= 1`.
Every value they let through reached `controller_dt` --
`(NUM_REGEN_FRAMES / fps) * generate_dt` -- which `MotionBricksPolicy` caches and
hands to the generator on every `get_actions` call. Measured with one config per
value: `fps=nan` gave a `nan` horizon, `fps=inf` gave exactly `0.0` (the
generator asked to integrate no time at all, under a success result), `fps=True`
gave `16.0` from a boolean, `fps=2.5` a fractional frame rate, and
`generate_dt=inf` an unbounded horizon. A numeric string escaped as a bare
`TypeError` from the comparison operator, naming neither the field nor the
config.

`speed_scale` had a second failure: the pair was normalised with `float()` and
only *then* range-checked, so `("1", "2")` was laundered into `(1.0, 2.0)` and
stored as if the caller had written floats, while `(nan, nan)` and `(1.0, inf)`
passed `lo <= 0 or hi <= 0 or hi < lo` untouched -- `nan` fails all three
comparisons. A scalar `speed_scale=0.5` raised `'float' object is not iterable`
out of the arity check itself.

`fps` now takes `positive_whole_number_error` and `generate_dt` and each
`speed_scale` component `positive_finite_number_error`, so the refusal is
identical word for word to every other rate and multiplier in the library rather
than merely equivalent in verdict. Pair arity is read with `sequence_length`, so
a scalar is refused by the arity message instead of raising past it. The
`min <= max` ordering rule stays local, since both components can be
individually usable and still be in the wrong order. A fractional
`generate_dt`, a NumPy scalar knob, a list-valued `speed_scale` and the upstream
defaults all stay first-class.
