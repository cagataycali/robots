### Fixed: a connection whose exchange did not complete is discarded rather than reused

`RemotePolicy._request` sends one message and reads one reply. When that read
did not arrive - the `request_timeout` deadline expiring on a slow remote VLA,
which is the ordinary reason that budget exists - the reply was still produced
and still queued on the socket, and nothing cleared `self._ws`. So
`_ensure_connected` short-circuited on the same connection and the next request
read the previous request's answer, and the one after that read the one before
it, for the life of the connection.

Neither layer could see the slip. `_request` inspected the reply only for
`MSG_ERROR`, and a stale action chunk carries exactly the `MSG_ACTIONS` type the
caller expects, so a robot executed a chunk computed for an observation it had
already moved past. A `MSG_OK` - what `reset`, `set_robot_state_keys` and
`set_control_frequency` are answered with - read as an action chunk became `[]`
through `reply.get("actions", [])`, which is indistinguishable from a policy
that chose to emit nothing.

`_connect` had the same shape one step earlier: it assigns `self._ws` before it
reads the handshake, so the two `ConnectionError`s it raises - a first frame
that is not `MSG_READY`, and a protocol version this client does not speak -
left a live socket cached behind a refusal. The mismatch was reported once and
the next request then served on the very connection the client had just
declared unusable, answering `[]`.

Both paths now discard the connection unless the exchange completed, through one
helper, and the next `_ensure_connected` opens a fresh one and replays the
pending config - the path that already existed for a connection that was never
opened. A `MSG_ERROR` reply is deliberately not a failed exchange: it is this
request's reply, the server marshals any dispatch failure back and carries on
serving, so the stream stays in step. The bookkeeping is a `finally` rather than
an `except` so a cancellation between the send and the receive, which leaves the
same undelivered reply behind, is covered too.

Discarding it moves the connection out from under a caller that had already been
told it was live. `_get_actions_blocking` asks `_ensure_connected` outside the
lock, so a second thread - the ordinary case here, an `rtc-prefetch` worker
beside the rollout thread - could pass that check and then wait on the lock while
the holder's read timed out and discarded the connection. It arrived in
`_request` with `self._ws` gone and hit a bare `assert`: an `AssertionError`
carrying no message, naming neither the connection nor the sibling that took it
away, and under `python -O` no assert at all, leaving `AttributeError: 'NoneType'
object has no attribute 'send'`. It now re-checks under the lock, as `reset`,
`set_robot_state_keys` and `set_control_frequency` already did; where they defer,
because the connect replay applies the config they carry, this one owes the
caller a chunk, so it opens a fresh connection and is served. `_request` states
the condition rather than asserting it, for any caller that does not re-check.
