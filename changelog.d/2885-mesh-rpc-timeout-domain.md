### Fixed: an unusable mesh RPC wait budget is refused instead of acted on

`Mesh.send` and `Mesh.broadcast` handed their `timeout` straight to a
`threading.Event` wait with no value domain. The `robot_mesh` tool already holds
that same parameter to `positive_finite_number_error`, and the docstring of the
tool's domain helper names these two methods as the consumer -- "every action
that reads it hands it to a `threading.Event` wait (`Mesh.send`) ... Only a
positive finite number can be honored". A caller reaching `Mesh` directly - a
test, a third-party integration, anything that imports it - got no such check.
That is the same tool-versus-library gap `send`'s own `validate_command` call
closes for the command, left open for the budget.

Every unusable spelling published the command first and only then failed. `nan`
and a negative make `Event.wait` return immediately, so the caller was handed
`{"status": "timeout"}` about 0.01ms after a command that did go out - the peer
may be executing it. `inf` and a string raised `OverflowError` / `TypeError` out
of methods contracted to return an envelope or a list of responses. `True` was
silently a one-second budget, and `None` waited forever, so the call never
returned.

`broadcast` carried the sharper consequence: the comment on its wait explains
that the window deliberately spans the full budget so an operator can tell "1 of
12 stopped" from "all stopped". An unusable budget returned at once and reported
an empty fleet for a broadcast that had been published.

Both methods now consult the shared domain before the turn is registered and
before `publish`, so nothing reaches the wire under a budget that cannot bound
the wait. `send` answers with its structured error; `broadcast` logs the reason
and returns no responses, which is how it already reports a client-side
rejection. A usable budget - including the fractional values the mesh suites
pass - is unchanged.
