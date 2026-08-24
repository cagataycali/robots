### Fixed: a validation guard no longer raises while rendering the refusal it is returning

The scalar guards in `strands_robots/utils.py` exist so a caller's bad value is
reported through a structured `{status, content}` result - every one of their
callers documents that result as the only channel a rejected value is reported
on. Building the message is part of that answer, and it was not a total
operation: the guards interpolated the caller's value directly, and rendering a
value can raise. So a guard could fail on precisely the path whose purpose is to
answer instead of raise.

Two values reach it, neither hypothetical. Rendering an `int` wider than
`sys.get_int_max_str_digits()` (4300 digits by default) raises `ValueError`, and
`device_connect`'s `@rpc()` surfaces forward a remote caller's number unchanged
while Python integers are arbitrary-precision. And a third-party type may raise
anything at all from its own `__repr__`: `numbers.Real` is a registration rather
than an inheritance, so a scalar that satisfies a guard's type test owes it
nothing else.

Measured across the family, the second case reached *every* guard except the one
already rewritten for the step surface - each of them raised on a value it had
already decided to refuse:

```
positive_count_error(-(10**5000), "n", "ctx")        # ValueError: Exceeds the limit (4300 digits)
tcp_port_error(10**5000, "port", "ctx")              # ValueError
entity_name_error("add_object", "name", 10**5000)    # ValueError
positive_count_error(Unprintable(), "n", "ctx")      # RuntimeError, from the value's own __repr__
```

The outsized-integer cells are narrower than they look, which is why they read as
clean: the guards that never convert raise only on a value they have *refused*,
never on one they accept.

Every one of them now renders through `_refusal_repr` or its new `str`
counterpart `_refusal_str`, which defer to `repr` / `str` wherever those work and
otherwise describe the value - by bit count for an `int`, since `int.bit_length`
needs no decimal conversion, and by type name otherwise. No verdict and no
agent-visible text changes: every message these guards already produced is
reproduced byte for byte, which is pinned rather than asserted.
`positive_whole_number_error` additionally renders on demand instead of building
its text on the function's first line, where it raised ahead of every verdict and
on the accept path too.

Two of the sites render a value plainly rather than quoted and needed the `str`
form rather than `repr`, because NumPy 2 reprs a scalar with its type and
converting would have silently turned `got 200.0` into `got np.float32(200.0)` in
text an agent reads. One of them, `camera_fov_error`'s open-interval branch,
looks unreachable by anything unrenderable - it runs only after
`math.isfinite(float(value))` has succeeded - but a registered `numbers.Real`
whose `float()` is finite and outside `(0, 180)` passes that test and is refused
there. The other, `validation_split_error`, renders a `total_tasks` read straight
out of a dataset's `meta/info.json`, so a JSON integer of any width reaches it.
Neither carries an `!r`, so both were invisible to a reading that looked for one.

The invariant is pinned by a scan of the module rather than a list of guards, so a
guard added later cannot reintroduce it silently. The scan keys on the parameter
annotated `Any`, which is how every guard here spells "the caller's value" - the
others are the `str` labels the call site supplies, which are literals and cannot
raise - and it reports the plain `{value}` form as well as `{value!r}`.

Two neighbouring escapes are deliberately left alone and pinned as out of scope.
The `float()` conversion in `positive_finite_number_error`, `finite_number_error`,
`camera_fov_error` and `positive_whole_number_error` raises `OverflowError` for an
`int` wider than a float, before any message is rendered; closing it needs a
decision this change does not, since `10**400` *is* a finite real number and "must
be a finite number" is the wrong reason to refuse it even though refusing is
right. And the container guards render a whole list, so one unrenderable element
takes down a refusal already decided - but this change's fallback is not the fix
there, because `<unrepresentable list>` erases every element that rendered fine
along with the element count, and the count is often the refusal's whole reason.
