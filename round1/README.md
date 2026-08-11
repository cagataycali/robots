# Round 1 - the classifier's handler width

`_renders_a_half_built_instance` turns a repr outcome into a verdict string the
survey compares against every class in the package. Catching `BaseException`
made an interrupt one of those verdicts. These scripts measure the three
candidate widths against nine outcomes and validate a mirror of the CodeQL rule
against the verdict GitHub published on the PR.

- `decision_table.py` - the three widths x nine outcomes, printed.
- `mirror_catch_base.py` - a validated mirror of `py/catch-base-exception`
  scoped to `tests/`; reproduces the published alert's line **and** column range
  (124, 5-33) on the pre-round head and reports 0 hits after.
- `capture.py` -> `facts.json` -> `compose.py` -> `handler-width-decision-table.png`.
  Every number the figure renders is asserted against `facts.json` first.
