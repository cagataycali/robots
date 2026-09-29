### Fixed: docs/learn/data/record.md states `add_frame`'s real contract

The page said `create` refuses a shape mismatch with `RecordingFrameError` and
names an action key it cannot record. `add_frame` raises `ValueError` for a
missing declared column, `RecordingFrameError` for a failed dataset write, and
drops undeclared action keys; the sentence now says so and a grader reads the
page, the docstring and the recorder together. Closes #4149.
