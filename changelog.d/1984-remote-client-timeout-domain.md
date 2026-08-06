### Fixed: a remote-inference client timeout that names no budget is refused rather than reported as an absent server

`RemotePolicy`'s and `LerobotAsyncPolicy`'s `connect_timeout` / `request_timeout` were
stored as given and handed to the transport - `websockets`' `open_timeout` and `recv` on one,
a gRPC call deadline on the other. Both constructors already refused their other numeric
parameters (`port`, `actions_per_chunk`), so these four knobs were the ones left behind.

The consequence was a misattribution rather than a late crash. Both clients wrap the first
transport failure in a `ConnectionError` naming the server and telling the operator to start
one, and `0`, a negative and `True` all fail inside that clause against a server that is
running and reachable - so an unusable timeout was reported as an absent server. `nan`, `inf`
and a numeric string escaped the clause instead, surfacing mid-rollout as an `OverflowError` /
`ValueError` / `TypeError` from library internals that named no parameter, because both clients
connect lazily on first use. Over gRPC, `0` is load-dependent on top of that: it succeeds
against a server that answers instantly and fails against one that takes a second.

All four now share `positive_finite_number_error` - the domain a control frequency, a rollout
duration and a bridge loop period already use - checked after `port`, so the narrower "this
address cannot be dialled" still wins when both are wrong, and before any transport import, so
the same mistake reports identically with and without the `[lerobot-async]` extra.

`inf` is refused deliberately rather than read as "no deadline", which is the one thing a
caller might mean by it: `websockets` raises `OverflowError` computing a deadline from it, and
gRPC reports `DEADLINE_EXCEEDED` immediately, making it indistinguishable from `0`. An
unbounded wait is not expressible through these knobs on either transport, so admitting the
value that looks like it says so would document a footgun instead of a budget. If it is wanted
later it needs its own spelling and its own decision.
