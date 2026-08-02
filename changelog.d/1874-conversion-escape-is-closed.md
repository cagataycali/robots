### Fixed: a numeric guard no longer raises on a value too large for a float

Four scalar guards in `strands_robots/utils.py` establish their domain by
converting the caller's value with `float()`: `positive_finite_number_error`,
`finite_number_error`, `positive_whole_number_error` and `camera_fov_error`.
`float()` raises `OverflowError` for a real whose magnitude exceeds
`sys.float_info.max`, so each of them raised instead of returning the structured
refusal their callers document as the only channel a bad value is reported on.
The conversion runs before any message is rendered, so the fix that made the
*rendering* total could not reach it.

```
positive_finite_number_error(10**400, "hz", "teleoperate")   # OverflowError
camera_fov_error("add_camera", "fov", 10**400)               # OverflowError
```

`int` is arbitrary-precision and `device_connect`'s `@rpc()` surfaces forward a
remote caller's number unchanged, so such a value is one request away. Two
properties of the escape are worth naming because they constrain the fix:

- **It is not an `int` problem.** `Fraction(10**400, 3)` is a registered
  `numbers.Real` that overflows identically, so the guard has to ask the question
  of the conversion rather than of the type.
- **It has two exception classes.** A `numbers.Real` registration guarantees no
  working `__float__`, so `float()` can raise `TypeError` here too - which is not
  a magnitude complaint and is not reported as one. Such a value is now refused
  with the text the guard already used for a value that is not a number at all.

Three of the four needed a new reason, because their own would have been a false
statement: `10**400` *is* positive, *is* finite, and *is* a positive whole number.
They now report that the value must be within the range of a 64-bit float. That
is not a new boundary - each guard already accepted up to `sys.float_info.max`,
`1e300` and `10**308` among them, and raised one step past it - so the range is
where the accepted domain already ended and all that changes is that the edge now
answers instead of raising.

`camera_fov_error` needed no new text. Its domain is bounded above at 180
degrees, so the interval message it already had is a true statement about a value
past the float64 range, and the overflow alone establishes that without a
comparison.

No verdict moved and no existing message changed: every value these guards
accepted is still accepted, every value they refused is still refused with
byte-identical text, and the whole effect is that 24 probes which used to raise
now return a string. That is measured against a control matrix per guard rather
than asserted.

One asymmetry is deliberate and recorded in both docstrings.
`positive_whole_number_error` refuses an outsized value where its sibling
`non_negative_whole_number_error` accepts one, although the two are documented as
the same policy with the floor moved. The difference is the consumer: MuJoCo's
`_MAX_STEPS_PER_CALL` bounds a step count with a reason of its own, while this
domain's callers include the mesh robots' `drive(count=...)`, which repeats an
actuation command - so an unbounded count there is an unbounded actuation loop
against a physical robot rather than a slow call.

The invariant is pinned by an AST scan of the module rather than a list of
guards, so a fifth converting guard cannot reintroduce it silently. The scan
reports every function whose `float()` conversion no `try` protects, and asserts
the set exactly: it is now the four container guards, which also carry this
escape (`finite_vector_error("raycast", "origin", [10**400])` raises
`OverflowError`) and are tracked separately, because closing it there needs an
elementwise message format - a whole-container fallback erases the element count
that is often the refusal's entire reason.
