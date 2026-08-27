### Fixed: the remote-policy connect replay holds the wire lock the other callers hold

`RemotePolicy` serialises its WebSocket with one lock, and its wire helper says so
in the section header above it: "call while holding `self._lock`". Four of
`_request`'s seven call sites did. The three inside `_connect` did not, and those
three run *after* `self._ws` is live - the handshake read, then the replay of any
state keys, control frequency and pending reset that were set before the
connection existed. The connect sequence is the widest wire user in the class and
it was the one outside the lock.

A second thread reaching the wire in that window overlapped the read, and
`websockets` refuses an overlapping read. Measured with a server parked mid-replay
and a client whose config was set before connecting: `reset(seed=7)` raised
`ConcurrencyError: cannot call recv while another thread is already running recv
or recv_streaming` - a report naming the transport's internals rather than
anything the caller passed, out of a call that never mentions threads. The same
window is reachable from `set_robot_state_keys` and from `get_actions`.

Two threads on one policy is the ordinary case rather than a contrived one. Every
policy coroutine resolves through `_async_utils`' reused worker thread, and the
async-RTC path in `simulation.policy_runner` submits prefetch inference to its own
`rtc-prefetch` worker while the rollout thread carries on stepping, so a
first connect and another call on the same policy land on different threads by
design.

`_ensure_connected` now holds `self._lock` across `_connect`. The unlocked fast
path stays, with the state re-checked under the lock, so two racing first-callers
open one connection rather than two; none of its three callers holds the lock, so
nothing deadlocks. A concurrent wire user waits for the replay and then proceeds,
which is what the lock was there to do. Reusing that lock rather than adding a
connect-only one is the point: a dedicated lock would make connects mutually
exclusive and still leave them racing `reset` and `get_actions`.

The precondition is now graded at runtime rather than by reading the source. A
witness lock records whether it was held when `_request` ran, so the coverage
`_connect` inherits from its caller counts, where a lexical scan would report
those three calls as unlocked either way. The pre-existing suite could not see
this: its threads run the *server*, and the client is driven from one thread
throughout.
