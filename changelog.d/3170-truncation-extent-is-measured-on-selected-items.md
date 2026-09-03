### Fixed: the truncation report measures both its numbers on selected items

`scripts/report_truncated_test_run.py` decides whether a pull request carries a
"the failure count is a lower bound, not a total" warning, from
`never_ran = selected - executed`. The two halves were read off different
populations, so the difference between them was not the number of items that
never ran.

`selected` came from a pattern that expected `deselected` and `selected` to be
adjacent on pytest's collection line. pytest appends those tokens conditionally
and in a fixed order (`TerminalReporter.report_collect`):

```
collected N items[ / E errors][ / D deselected][ / S skipped][ / X selected]
```

A collection error or a module-level skip lands between them, the `selected`
group reads as absent, and the count falls back to the pre-deselection total.
`executed` summed the outcome labels on the summary line, which include the ones
*collection* produced -- a module that fails to import is an `error`, one that
skips itself for an absent optional dependency is a `skipped` -- and neither was
ever a selected item.

The two errors point opposite ways and partly cancel, which is why a run whose
tokens happened to be adjacent read correctly and the existing pin, whose
fixture poses exactly that arrangement, passed.

Measured on five real pytest sessions, one per arrangement of those tokens,
against the counts pytest's own session held. A run that selected 2 items with
`-k` and ran both, in a tree where one module skipped itself, was reported
`truncated` with 2 of 5 items never run -- **40% of a session that finished
everything it selected**. Four of the five arrangements misreported the extent.

Both numbers now come from the collection line that already reports the
deselected and collection-phase counts, and `tests/session_truncation.py` --
which holds `len(session.items)` and counts the items it saw entered -- settles
the extent whenever its section is in the log, so one derivation of the
subtraction owns it and the text arithmetic is the fallback. The report says
which of the two produced the numbers.
