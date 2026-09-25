### Fixed: a warn-once memo is emptied at the test boundary, not by whoever was bitten

Sixteen modules report a posture once per process and remember the keys they
have already reported, so the first test to spend a key decided that every later
test reading that report got silence. Thirty-three test modules reset one of
those memos at sixty-one sites - each only the memo it had been bitten by, and
two of them restoring the keys they found spent. `tests/conftest.py` empties all
sixteen now; a reset between two asks inside one test, which grades the
once-per-process gate itself, stays. `_reset_posture_warnings` and
`_reset_resolution_warnings`, two private helpers that existed only for that
isolation, are gone with their last callers.
