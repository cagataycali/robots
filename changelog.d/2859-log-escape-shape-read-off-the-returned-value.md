### Tests: the log-escape's shape is read off the value the caller receives

`strands_robots.policies._log_safety.sanitize_log_value` escapes the two line-break
characters as chained `.replace` calls with literal arguments rather than a loop over
a table of pairs, and the spelling is load-bearing: `py/log-injection`'s only barrier
reads the call site rather than its effect, holding for a `.replace` whose first
argument is a *string literal* equal to `"\r\n"` or `"\n"`. A cell holds that shape so
tidying the function back into a loop fails rather than silently restoring the report
on every sink the helper is the only escape for.

That cell walked the whole function body for such a call, which is satisfied by an
escape whose result never leaves the helper. Computing the chain into a local and
returning the raw text passes it while the record still splits, so the cell reported
true about a shape that was false; the forging cells caught the mutation, but under
names that describe a split record rather than the shape the cell exists to hold.
Reading the calls off the single `return` makes the cell's own name true.

Two cells sit beside it. A premise states the accepted set the rule reads and that
`"\r"` is not in it, which is what makes the second link in the chain a decision
rather than a leftover -- a payload can arrive carrying a bare carriage return, so the
escape has to cover a spelling the rule does not recognise. An over-reach control
holds the rendered text against the table loop the chain replaced, exhaustively over
every string up to four characters drawn from `\r`, `\n`, a backslash and an ordinary
letter, so "the two forms render identically" is measured rather than asserted; a link
that keeps its literal argument and renders the wrong escape fails it while the shape
cells stay green.

The shape cell's docstring carried real line breaks inside the inline-code spans where
those two spellings belonged, so the sentence naming them rendered with the spellings
invisible -- in a module whose whole subject is that substitution. They are written as
escapes now, and the docstring body is de-indented to the level the rest of the file
uses.
