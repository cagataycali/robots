### Added: `DDSPublisher` for the G1 write path (write half of the DDS engine)

The G1 native driver's DDS engine has grown the sibling class the write
path needs. `DDSSubscriberSet` already funnels the read path through
`_DDS_INIT_LOCK` and a lazy SDK import; `DDSPublisher` mirrors both, and
adds a `(topic, message_class)` cache so a control loop asking for the
`rt/lowcmd` publisher every step constructs it once:

```python
from strands_robots.tools.g1._dds_engine import DDSPublisher

pubs = DDSPublisher("eth0")
pubs.start()                              # ChannelFactoryInitialize under the lock
err = pubs.publish("rt/lowcmd", LowCmd_, message)   # None on success, string on error
```

The class does not import `unitree_sdk2py` at module load, so the driver
still loads on Thor and CI. That is the invariant the shared lock and the
lazy import together buy: one construction lane for the entire G1 DDS
layer, and no SDK on module load.

This ships the transport primitive that issue #361 (send_action wired) and
its follow-up (control loop at 500Hz) will consume; the driver itself
still refuses motion until those PRs land.
