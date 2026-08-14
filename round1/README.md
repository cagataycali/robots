# Round 1 - py/unused-local-variable, and the vacuous assertion it exposed

Scripts backing the Round 1 claims on strands-labs/robots#2255.

## `mirror.py`

A mirror of `py/unused-local-variable`, validated against the two alerts GitHub
published on head `b668e79d` - both reproduced at the exact line **and column
range**, which is what makes its verdict on the fix trustworthy rather than a
re-implementation that might differ.

```
PYTHONPATH=<repo> python3 round1/mirror.py tests/tools/test_teleop_auto_accept_reports_failure.py
```

Reports no findings on head `3a87b561`. The PR's code-scanning alert list is
empty on the same head, so the mirror and the scanner agree.

## `mutate.py`

The round's mutation table. Each anchor is AST-scoped to its enclosing function
and asserts `in_range == 1` before replacing, and the source is restored
byte-identically in a `finally`.

| mutation | new module | pre-existing |
| --- | --- | --- |
| unmutated control | 0 | 0 |
| drop the session name from the record | 1 | 0 |
| revert the unused-local fix | 0 (scanner-only) | 0 |

The first row measured **0** before this round: with the session named `"cal"`
and the record's own prose reading `"calibration"`, `assert "cal" in message`
was satisfied by the wording whether or not the session was ever named. The
third row is behaviour-preserving by construction, which is why it needs the
scanner rather than a test.
