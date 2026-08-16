### Fixed: a ZMQ inference timeout that names no wait budget is refused rather than reported as an absent sidecar

`Gr00tInferenceClient` and `MoveIt2InferenceClient` each took a `timeout_ms`,
stored it verbatim, and handed it to `setsockopt(RCVTIMEO)` and
`setsockopt(SNDTIMEO)`. Neither checked it. Both now share the
`utils.coerce_zmq_timeout_ms` domain, so an unusable budget is refused at
construction with a `ValueError` naming the class, the parameter and the bound.

This is the concern #1984 settled for the WebSocket and gRPC clients, on the
third remote-inference transport. It was missed there because that change
enumerated the knobs by name - `connect_timeout` / `request_timeout` - and these
are spelled `timeout_ms`. `MoveIt2Policy` already refused the other numeric
parameter it owns, `port`, through `tcp_port_error`, and forwarded `timeout_ms`
unguarded eight lines later.

The consequence was a healthy sidecar reported as unreachable, and it was
quieter than the sibling case. Both clients' `ping()` catches every exception and
returns `False`, logging the reason at `debug`, so there was no signal at default
log level at all - where the WebSocket client at least raised a `ConnectionError`
with a (misleading) remedy. Measured with the real client classes against a real
loopback REP sidecar that answers correctly:

| `timeout_ms` | before | `ping()` vs a healthy sidecar | after |
| --- | --- | --- | --- |
| `15000` (default) | accepted | `True` | unchanged |
| `0`, `False` | accepted | **`False`** - ZMQ's "return immediately" spelling, so every request raises `zmq.Again` whether or not a sidecar exists | refused |
| `True` | accepted | **`False` on MoveIt2, `True` on GR00T** - a silent 1 ms budget, so the verdict depends on how long the peer takes | refused |
| `-1` | accepted | **never returns** | refused |
| `-2`, `-15000` | bare `ZMQError: Invalid argument` | never reached | refused |
| `nan`, `inf`, `1.5`, `'15000'`, `None`, `[15000]` | bare `TypeError` from inside pyzmq, naming no parameter | never reached | refused |
| `2**31`, `10**400` | bare `OverflowError` | never reached | refused |
| `15000.0`, `np.int64(15000)` | **bare `TypeError`** - a usable budget the transport refuses | never reached | **accepted, stored as `int`** |

The last row is why this is a fix rather than only a refusal. `setsockopt` takes
a C `int` and rejects every other spelling of the same budget, so a timeout read
out of a JSON config or a config array could not be used - while the sibling
transports accept exactly those spellings. `coerce_zmq_timeout_ms` returns the
value coerced to `int`, so one configured budget is no longer usable on two
transports and unusable on the third.

`True` deserves its own note: it is an `int` subclass, so it was not merely
accepted but *load-dependent*. It passed against the GR00T sidecar and failed
against the MoveIt2 one on the same machine, purely on serializer cost - the
shape that passes a smoke test and fails under a real checkpoint.

`-1` is the one value with a real claim, and it is refused deliberately. ZMQ
documents it as "block forever", so unlike the `inf` of the sibling transports it
*is* honoured - which is what makes it dangerous rather than useless. It
reinstates on the request path exactly the unbounded hang that `LINGER = 0`, set
two lines below in the same `_init_socket`, was added to prevent on teardown, and
it leaves `ping()` - whose whole contract is to answer `True` or `False` about
connectivity - unable to answer at all. Neither client documented it as a
spelling. An unbounded wait on a robot control path wants its own parameter and
its own decision rather than arriving as a negative millisecond count.

The ceiling is the transport's, not a policy choice: `RCVTIMEO` is stored as a C
`int`, so `utils.MAX_ZMQ_TIMEOUT_MS` is `2**31 - 1` ms (close to 24.9 days) and
one millisecond more raises `OverflowError`. This is also why
`positive_whole_number_error` could not be applied on its own - it accepts
`2**31` as a positive whole number inside the float64 range. Its docstring
claimed "No consumer of *this* domain owns one" of these ceilings; that is no
longer true and is corrected in place.

The accepted budget is read from the caller's value exactly once. The first
spelling of the helper validated with `positive_whole_number_error` and then read
`value` twice more - `float(value)` for the range and `int(value)` for the result
- on the reasoning that the guard had made those conversions safe. That is the
reasoning #1875 shipped for the vector coercions and #1906 withdrew, and
`utils.py` carries a module-wide scan asserting that no function in it converts
with a `float()` no `try` protects, against a set that is empty and is asserted
so it "can neither grow nor be quietly narrowed". So the ceiling is now compared
against the `int` the helper itself produced: `int` is exact for every value that
reaches it (the guard has established a finite, integral `numbers.Real`) and is
arbitrary-precision, so an integral `1e300` converts rather than overflowing and
is then refused by the ceiling. A new guard joins that invariant rather than the
exception list.

Each guard is placed ahead of its constructor's `_load_zmq()`, so the same caller
mistake reports identically on an install with the `[groot]` / `[moveit2]` extra
and one without it, and a refused budget leaves no socket configured behind it.

Every value that already named a budget still does: the default `15000` is
unchanged and the usages across the tests, docs and examples all sit inside the
accepted domain.

Pinned by `tests/test_zmq_timeout_ms_domain.py` - 130 cases, 70 of which fail
with the guards removed. `pyzmq` is imported optionally rather than through a
module-level `importorskip`, because both clients load it lazily and refuse an
unusable budget *before* that call: on an install without the `[groot]` /
`[moveit2]` extra 94 of the cases still run and 60 of them still fail with the
guards removed, where a module-level skip would have taken the structural drift
guard with it. It asserts the misattribution rather than the raise (an
unusable budget can no longer report a live loopback sidecar as unreachable), the
two clients' verdicts are asserted equal over the whole probe set so they cannot
diverge, the ceiling is asserted against `setsockopt` rather than restated as a
constant, and premise tests pin pyzmq's own treatment of `0`, `True`, `-1`, `-2`
and the non-`int` spellings so the reasoning above fails loudly rather than going
stale. A structural check requires every module that sets `RCVTIMEO` / `SNDTIMEO`
to route the value through the shared domain, so a third ZMQ client cannot ship
without joining the rule. The read-once property is asserted as a delta against
`positive_whole_number_error` called alone, so the guard's own reads are the
baseline and what the coercion adds is measured rather than the guard's internals
restated; the three cases covering it fail on the first spelling.

The other unvalidated timeout surfaces #1984 named and left - `IotTransport.connect_timeout`,
`VeraConfig.server_ready_timeout`, `RosBridgeRobot.navigate_to`'s `timeout` - stay
out of scope for the reasons it gave, and are still their own change.
