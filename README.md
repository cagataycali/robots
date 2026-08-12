Measured artifact for the IotMqttTransport client-teardown fix.

* `measure.py` - drives the four `self._client.stop()` call sites with a client
  whose `stop()` raises and reports, per site, whether the failure escaped, what
  was left behind, and what was logged.
* `capture.py` - the same sweep, structured for the figure; run once per tree.
* `mutate.py`  - applies 5 plausible reversions and runs both test arms.
* `compose.py` - builds the figure; asserts every rendered number against the
  two JSON dumps before saving.
