### Fixed: `IotMqttTransport` refuses a connect timeout it cannot spend

`IotMqttTransport(connect_timeout=...)` stored whatever it was handed and spent it
on `threading.Event.wait` inside `connect()`. It is the third surface in the
library to carry that parameter name; the two remote-inference clients
(`RemotePolicy`, `LerobotAsyncPolicy`) already refuse a value that names no wait
budget, and the mesh transport did not.

An unusable value was worse than a late crash because `connect()`'s report could
not be told apart from a broker that was genuinely unreachable. Measured against
a fake broker whose CONNACK arrives 50 ms after `start()`: `0`, `-1` and `nan`
made the wait return in ~0 ms, so `connect()` logged "IoT connection to ... timed
out after 0.0s", stopped the client that was connecting normally and returned
`False` - pointing the operator at the endpoint, the certs and the broker, the
three things that were not wrong. `inf` and `'15'` raised `OverflowError` /
`TypeError` from the wait, which sits after the `try` that contains client
construction, so they escaped a method documented to return `bool` leaving the
MQTT5 client started with no `stop()` on any path. `None` blocked forever while
`connect()` held the instance lock, so `close()` could never run, and `True`
was a silent one-second budget.

The constructor now applies `positive_finite_number_error` - the same domain, the
same message shape and the same reasoning the two clients record. The guard sits
in the constructor rather than beside the wait so that it also precedes the
`awsiot` import inside `connect()`: the same mistake reports identically with and
without the `[mesh-iot]` extra installed. A structural test scoped by the
parameter name keeps a fourth surface from shipping the knob without the rule.
