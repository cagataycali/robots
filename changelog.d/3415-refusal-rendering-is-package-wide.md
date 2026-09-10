### Fixed: a guard anywhere in the package answers a value it cannot render

`strands_robots.utils` established the rule that a guard must not raise while
building a refusal: rendering a value can fail - `repr` of an `int` wider than
`sys.get_int_max_str_digits()` raises `ValueError`, and `numbers.Real` is a
registration rather than an inheritance, so a scalar that satisfies a type test
owes the guard nothing else - and a guard exists precisely so a bad value is
reported through a returned result instead of an exception.

The scan that established the rule read one file, and the same guard shape is
written in nineteen other modules. Forty-three functions across twenty modules
interpolated the caller's value straight into their message, and twenty-one of
them raised when handed a value they had already decided to refuse:
`randomization_range_error`, `finite_non_negative_error` and six more in
`simulation/base.py`, all five compositor bounds, three of the MJPEG stream
guards, the ray-batch and excluded-body coercions, and both training interval
checks. `_closed_unit_interval_error(Fraction(-10**5000, 10**5000 + 1), ...)`
raised `ValueError` out of a refusal it had already decided - stdlib only, no
registered type and no hostile `__repr__`.

Every one of them now renders through `refusal_repr`, `refusal_str` or
`refusal_container_repr`, which defer to `repr`/`str` wherever those work, so no
verdict and no existing message text changes. The three renderers are package
API rather than `utils`-private helpers, because the contract they serve is
stated over the package: `tests/test_refusal_messages_never_raise.py` now scans
every module instead of one, which is what closed the other twenty modules.
