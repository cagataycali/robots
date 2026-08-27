### Fixed: a non-finite direction goal is refused, and the refusal names the key it came from

`MotionBricksPolicy` resolves two of the issue #300 well-known goal keys -
`target_velocity` and `target_heading` - through one reader that normalises each
to a unit direction in the world XY plane. That reader falls back to "walk
straight ahead" for a command with no direction, and its own docstring gave the
reason: so an all-zero command "does not produce a NaN direction".

The fallback is a magnitude test, so it could not cover a non-finite component.
`nan < 1e-6` is `False`, and `inf` divided by its own norm is `nan`, so either
one fell straight through it and produced exactly the outcome the fallback exists
to prevent: `target_velocity=[nan, 0.0]` resolved to a `movement_direction` of
`[nan, nan, nan]`, handed to the generator by a call that reported success. The
same held for `target_heading` and the `facing_direction` it drives. The
per-component domain is now the shared `finite_vector_error` the simulation
setters already hold their own vectors to, so a `nan`/`inf` component, a
non-numeric element and a `bool` are each refused in the library's own words
rather than read as `1.0` or surfaced as numpy's `could not convert string to
float`.

The refusal named neither key. One reader serves both, and both answered
`direction vector must have 2 or 3 entries`, so a caller passing both could not
tell which one had been refused. Each call site now passes the key it reads. The
sibling locomotion family already did this - `WBCPolicy._validate_velocity`
answers `target_velocity must have at least 3 elements [vx, vy, omega]` - and the
policy contract's stated reason for leaving the component count out of the goal
vocabulary is that "each receiver states its own arity and refuses a shape it
cannot use", which needs the refusal to say which receiver and which key. The
mesh wire validator defers the count here deliberately ("the component COUNT is
not checked against any receiver's arity here"), so this reader is the only place
it is checked at all.

And the message stated a rule the check did not enforce. Five statements in the
package gave the accepted shape as two or three components - the reader's
docstring, the module docstring, the constructor's `target_velocity` entry, the
refusal text itself and `docs/policies/motionbricks.md` - while the check tested
the lower bound alone. A four-, six- or twenty-component vector was accepted and
read for its first two entries, so a caller who packed a six-component spatial
twist got planar motion and no error. The count is now the closed range every
one of those statements already described.

The constructor's `target_velocity` default went through no domain at all and was
injected into every call that passed none. It now goes through the same reader,
so a default that cannot be honoured is reported at construction rather than on
the first step of a started rollout - the ordering the sibling family's own
constructor comment gives as its reason.

Nothing caught any of this because the suite stated the property without driving
the input that violates it. Two shipped cells say so in their own words - "the
builder rejects it instead of fabricating a NaN/garbage heading", "an all-zero
command must not yield a NaN direction" - and both pass only the inputs where the
fallback does fire. Both have been narrowed to the half they grade and now point
at the new cells for the other.
