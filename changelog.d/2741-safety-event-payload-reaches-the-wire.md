### Fixed: a safety event carrying a reading reaches the wire instead of being dropped

`SensorLoopsMixin` builds nine records addressed to a `strands/<peer>/...` topic. Eight of
them -- every `_read_*` reader -- pass the record through `_coerce_record` before it is
published, because a sensor pipeline reports its readings as numpy and `json.dumps` refuses
a `float32` (#2730). `publish_safety_event` builds the ninth, and did not coerce it.

The consequence is total rather than partial: `_put_zenoh_directly` encodes the payload
before it reaches the wire, so the event was never published at all. It is not a transient
failure the next call retries either -- every call built the same way fails identically.
Meanwhile the audit half of the *same* call wrote a `sig="SERIALISE_FAILED"` poison record
and logged at ERROR, so the two halves of one call reported two different things: a forensic
trail asserting that a safety event was raised, and no peer having received it.

That disagreement is the one `session._report_unencodable_payload` describes in its own
docstring. #2638 raised the report from DEBUG to ERROR so an operator could at least see the
drop; the wire half still published nothing, and `_coerce_record` did not exist yet.

Which readings were lost depends on the numpy *width*, which is why the gap was easy to
miss. Measured through the encoder for a payload of `{"v": <value>}`:

| value | before | after |
|---|---|---|
| `np.float32(2.97)` | dropped (`TypeError`) | published |
| `np.int64(3)` | dropped (`TypeError`) | published |
| `np.bool_(True)` | dropped (`TypeError`) | published |
| `np.array([1.5, -2.5])` | dropped (`TypeError`) | published |
| `np.float64(2.97)` | published | published |
| `2.97` | published | published |

`np.float64` subclasses Python's `float`, so a payload built from one always encoded and the
same code reading a `float32` dropped every event. A reading is exactly what a safety
payload carries -- the joint value that tripped a limit, the distance that closed -- and the
public method's own docstring named no payload contract, while its audit sibling
`log_safety_event` documents "Must be JSON-serialisable".

The payload is now coerced once and the same coerced mapping is handed to both halves, so
they report one event rather than two. Coercion happens in a copy: the caller keeps the
mapping they passed unedited, and a fire-and-forget publish no longer holds an object the
caller can still mutate after the call returns. Nothing else about the event changes -- the
wire severity is still uniformly `"info"` (#272), the real severity still reaches the audit
record only, a stopped host still publishes nothing, and an audit-write failure is still
survived.

The boundary is that coercion repairs readings rather than laundering payloads. A value that
is genuinely not a reading is still passed through untouched, so the transport still reports
it by name -- substituting a repr would publish a record that misstates what the safety path
saw, which is worse than a reported drop.

`tests/mesh/test_safety_event_payload_reaches_the_wire.py` drives the method through a host
that *encodes* what it is handed, which is what the pre-existing coverage did not: the
`publish` double in `tests/mesh/test_sensor_readers.py` records the payload, and the encoder
runs one layer down in the transport, so a recording double accepts a payload the wire never
would. Each row carries a premise that its raw value is what the encoder refuses. The rule
that every record bound for the wire is coerced is derived from the module -- a method
building a dict literal with a `peer_id` key is building one -- so a publisher added later is
held to it on arrival; planting a tenth uncoerced builder fails the derived pin and passes a
hardcoded one. Eleven plausible regressions were applied: nine fire a different set and the
control is clean.
