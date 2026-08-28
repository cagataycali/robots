### Fixed: the Microduck transport knobs are refused rather than handed on unexamined

`MicroduckDriver.__init__` takes three numeric knobs. Two of them reached a
consumer that cannot report what it was given, and the driver's own actuation
flags already follow the opposite convention: `active` and `enable_torque`'s `on`
go through `boolean_flag_error` because reading them by truthiness would be
silent wrong actuation. The transport knobs were the pair left outside it, and
the sibling constructors both hold theirs to a shared domain - `G1Driver` checks
`battery_floor_pct` with `finite_number_error`, and `ReachyDriver` refuses an
unusable `api_port` before it has a daemon to address at all.

`timeout` is handed to `socket.settimeout` and to the reply wait. Four of eight
spellings raised out of `connect_eagerly` - a method declared `-> str | None`,
whose whole contract is to name what went wrong - from inside the socket call,
identifying neither the driver nor the parameter: `nan` and `-1.0` gave
`ValueError`, `inf` an `OverflowError`, and `"5"` a `TypeError`. Reachable
through the public route: `Robot("microduck", mode="real", timeout=float("nan"))`
builds, then `connect_eagerly()` raises `ValueError: Invalid value NaN (not a
number)`. The two that did *not* raise were worse. `True` set a silent one-second
timeout, and `None` put the socket in blocking mode and made the reply wait
unbounded - removing the only bound the parameter exists to impose. `0.0` did not
time out either; it made the socket non-blocking, so an absent daemon was
reported as "did not answer" for a reason that was not the one that happened.

`subscribe_hz` is interpolated straight into the `robot.subscribe` params. Driven
against the mock robotd over a real socket, `nan` and `inf` put
`{"hz":NaN}` and `{"hz":Infinity}` on the wire while `connect_eagerly` returned
success. Those are not JSON: RFC 8259 has no such literals, and a strict parser
refuses the frame, so the driver reported an established connection having sent a
subscribe robotd cannot read. `True` sent `{"hz":true}`, `2.5` sent `{"hz":2.5}`
and `-5` sent `{"hz":-5}`, all where an integer decimation is expected, and a
`numpy` integer is not JSON-serialisable at all - `json.dumps` raises on it.

Both are now checked against the shared domains, `positive_finite_number_error`
and `positive_count_error`, and refused at construction before any attribute is
recorded, so a refused build leaves no half-made driver behind. The strict-int
member is the right one for `subscribe_hz` precisely because robotd is sent an
integer: it is what rejects the float and the `numpy` scalar that would otherwise
reach the serialiser. `subscribe_hz=None` keeps its documented meaning of every
control tick, and the usable spellings are unchanged - the default still sends no
`hz` key and `subscribe_hz=30` still reaches the wire as `{"hz": 30}`.

The third knob, `api_version`, is deliberately left without a domain, and the new
suite records the measurement behind that rather than asserting it: a `nan`, a
`True` or a `"1"` is compared against the Hello reply and already produces the
named version-mismatch refusal, so a constructor check would only restate a
refusal the handshake already gives.
