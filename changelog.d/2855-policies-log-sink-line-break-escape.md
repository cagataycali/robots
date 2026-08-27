### Fixed: an observation-derived value can no longer split the log record that quotes it

Three of the eight `py/log-injection` sinks issue #2853 counted could forge a log
record, and five could not, and the difference was not the one the rule reports.
What decides it is whether the value can still carry a `\r` or `\n` at the moment
it is interpolated. `_to_lerobot_observation`, `_build_observation_batch` and
`_resolve_camera_targets` each hand a bare `str` -- a camera name, a language
instruction slice -- to a bare `%s`, so a line feed in it put the tail of the
warning on a line of its own, where a `FileHandler` or a shipper that frames on
`\n` reads it as a record this process never wrote, complete with whatever
timestamp and level the payload chose to spell.

The other five were escaped, but incidentally rather than by construction: the
two state-key sinks interpolate their keys inside a *list*, and `repr` escapes a
`str` element, while the cuRobo and MoveIt2 state sinks reach `%r` only after
`tolist()` has run. Nothing at those sinks said so, and the property is one
readability edit away from gone -- render the same keys as `', '.join(keys)`, or
let a list hold one object whose `__repr__` spans lines, and the break is back on
the wire with no sink-side change to notice it. That last case is what the two new
cuRobo/MoveIt2 cells drive, and both fail on the pre-change tree.

So `strands_robots.policies._log_safety.sanitize_log_value` now escapes the two
break characters at all eight sinks, and the property is stated where the record
is emitted instead of being inherited from the caller's formatting choice. It
touches `\r` and `\n` only: a key list, a joint-state repr and a remedy sentence
are mostly brackets, quotes and commas, and a broader filter would corrupt the
diagnosis the message exists for. Each break becomes its own visible two-character
escape rather than being dropped, which is what `repr` would have shown, so a
joint whose name genuinely contains a newline stays diagnosable. Escaping is
idempotent on already-escaped text, so the five sinks that were incidentally safe
render byte-identically to before.

Three sinks changed a format spec from `%r` to `%s` over an explicit `repr()`,
which renders the same bytes `%r` rendered. `tokens.shape` at the tokenizer sink
is deliberately left unwrapped: it is a shape the policy computed, not anything
the observation supplied, and wrapping it would state a provenance it does not
have.

The escape is written as two chained `.replace()` calls with literal arguments
rather than a loop over a table of pairs, and the spelling is load-bearing rather
than a style choice. `py/log-injection`'s barrier in CodeQL's Python pack
(`ReplaceLineBreaksSanitizer` in
`semmle/python/security/dataflow/LogInjectionCustomizations.qll`) holds only for a
`.replace` call whose first argument is a *string literal* equal to `"\r\n"` or
`"\n"`; a name read from a constant satisfies neither conjunct, so a table-driven
loop escapes the break exactly as well while being recognised as no barrier at
all. The rendered text is identical either way -- and order-independent, since
neither escape contains a raw break, so the two replacements cannot overlap -- but
`"\r"` alone is not in that literal set, so `"\n"` goes last and the value leaving
the helper is the result of the call the barrier recognises. A cell asserts the
first arguments are literals, reading them off the *returned* expression rather
than the function body, because a `.replace` whose result is discarded escapes
nothing the caller receives -- so tidying the function back into a loop fails
rather than silently restoring the report. That the rendered text is unchanged is
held exhaustively over every string up to four characters drawn from `\r`, `\n`, a
backslash and a letter, compared against the table loop the chain replaced.
