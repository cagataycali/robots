### Fixed: a vector guard must not raise when the value's own iteration fails

`finite_vector_error` asked whether its value was iterable with `iter(vec)`
inside `except TypeError`. `TypeError` is what "not iterable" means in Python and
the only exception a well-behaved `__iter__` raises, so a `__iter__` raising
anything else propagated straight out of the guard - and out of the structured
`{"status": "error"}` tool-result contract the guard exists to keep:

```python
class HostileIteration:
    def __iter__(self):
        raise RuntimeError("no iteration for you")

finite_vector_error("raycast", "origin", HostileIteration())   # RuntimeError
```

This was the fourth and last escape of one shape - a guard whose entire purpose
is to answer an unusable input with a message, raising instead - after the
rendering escape, the scalar-conversion escape and the container
conversion-and-rendering escape. `coerce_size_vector` inherited it, being the
only other surface that reaches this guard's `iter()`; the other container guards
take `len()` or test `isinstance(value, Sequence)` first and never reached an
iteration to escape from.

Such a value is now refused with its own text, naming the exception that stopped
the iteration:

```
raycast: 'origin' could not be iterated: RuntimeError: no iteration for you
(got <unrepresentable HostileIteration>). Pass a list or tuple of numbers.
```

The wording is new rather than reusing the existing `must be a list/tuple of
numbers`, because the two are not the same measurement: a value whose iteration
raised may well have held numbers, and the guard never found out, so reporting
the domain verdict would state something it did not measure. Every verdict that
existed before is unchanged, including the `TypeError` branch that answers a
plain non-iterable.

The exception's own text is rendered through the same `_refusal_str` helper as
every other refusal, since a value hostile enough to raise a non-`TypeError`
from `__iter__` is not one whose exception is assumed to have a working
`__str__` - that would reintroduce the rendering escape inside the fix for this
one. The probe also now binds the iterator it checked and iterates that, rather
than calling `iter(vec)` for its exception and iterating `vec` again, so the
`__iter__` that was checked is the one that runs.
