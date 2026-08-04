### Fixed: `robot_mesh` bounds the two numeric options its command body cannot carry

`robot_mesh`'s `duration` and `policy_port` ride inside the command body that
`validate_command` inspects, so that validator already bounds them. `timeout`
and `limit` never enter a command body and had no domain at all.

`timeout` becomes a `threading.Event` wait, which returns immediately for `0`,
a negative value or `nan` - so the tool reported `{"status": "timeout"}` under
`status="success"` for a peer it never gave the chance to answer, having waited
0.0s of the 30s a caller asked for. `stop`'s `min(timeout, 5.0)` cap did not
help, because `min(nan, 5.0)` is `nan`. `inf` surfaced an `OverflowError` from
the deadline arithmetic, `True` was a silent 1s budget, and a string or `None`
reached a bare comparison; `None` blocked forever.

`limit` is a slice index into the `inbox` buffer, and `inbox` is the action that
pulls a peer's stream into an agent's context. `0`, `-5` and `nan` all selected
the *whole* buffer rather than capping it, and `2.7` / `"50"` / `None` raised
`TypeError` out of a dispatcher documented never to raise.

Both are now checked against the shared domains - `timeout` against
`positive_finite_number_error` (the same domain the ROS transports' `timeout`
uses, being the same quantity consumed the same way) and `limit` against
`positive_count_error` - before the human-in-the-loop approval gate, the
rate-limit accounting and any transport call. Only the options an action reads
are checked, so `peers` is never refused for a wait budget it does not consume.
