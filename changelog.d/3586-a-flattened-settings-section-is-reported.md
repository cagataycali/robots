### Fixed: a settings section given a scalar is reported, not dropped in silence

`dashboard.settings` gives a caller two doors for grading a patch: `unknown_keys`
reports NAMES the schema does not know, `update_strict` reports VALUES it cannot
use. Between them they are meant to account for every part of a patch that does
not land, because the only other thing a caller can report is the changed list --
and an empty changed list with nothing else to say reads as a success.

One shape escaped both. `_update` skipped a section whose value was not a mapping
in the same clause it skipped an unknown section (`if section not in _SCHEMA or
not isinstance(values, dict)`), so `{"security": "<token>"}` -- written instead of
`{"security": {"auth_token": "<token>"}}` -- returned no unknown name, no error
and no changed key, and stored nothing. `unknown_keys` skipped it too, by design,
since the name is one the schema knows. The store already distinguished the case:
`{"bogus": "x"}`, an *unknown* section flattened the same way, was reported by
name. Only the section it recognised was dropped.

`security` carries `auth_token`, the bearer every `/api` and `/ws` request must
present, so on that section the silence has a direction: the operator is told the
write went through and the door stays open. The flattening is the ordinary
mistake -- a hand-written request body, a config file authored by hand, a UI that
sends one field of a section rather than the section.

The two clauses are now separate. A known section given something other than a
mapping of its keys is an unusable VALUE, so `update_strict` reports it under the
section's own name, in the wording the module already uses for a value of the
wrong shape (`security: expected a mapping of security keys, got str`). The
lenient path (`update`, the file, the environment) has no error channel to report
through, and there is no key whose shape a degrade could fall back to, so it
still skips -- at WARNING, because there the report the caller can make is a
success, which makes the log line the only signal the patch went nowhere.

`unknown_keys` is unchanged and now states the split: names are its answer,
values are `update_strict`'s, so neither door has to guess what the other
reported.
