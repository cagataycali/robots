### Fixed: closing the G1 DDS engine releases its endpoints

`DDSSubscriberSet.close` and `DDSPublisher.close` dropped their references and
relied on garbage collection, each justifying that with a docstring asserting
that the `unitree_sdk2py` class "has no explicit close". Both classes do define
`Close()`, and for a subscriber the collection half is not merely optimistic but
impossible: `subscribe()` builds every subscriber with `queueLen=10`, and at any
queue length above zero the SDK starts a `ch_reader` daemon thread whose target
is a bound method of the channel's reader. The running thread keeps the channel
reachable, so the CycloneDDS `__del__` that would release the `DataReader` never
runs.

Measured against `unitree_sdk2py` 1.0.1, one `ch_reader` thread per
subscription:

```
after subscribe:              1
after del + 3x gc.collect():  1     <- what close() used to do
after an explicit Close():    0
```

So `G1Driver.cleanup()` released nothing. The decoder callbacks kept filling
caches for a driver reporting itself disconnected, and because `cleanup()` then
sets `_connected = False`, the documented re-attach route - `cleanup()` followed
by `connect_eagerly()` - subscribed all four topics a second time. That is the
duplicate subscription the engine's shared construction lock exists to prevent,
and it is the hazard `connect_eagerly()`'s own docstring already names when it
declines to rebuild a live subscriber set.

Both `close()` methods now call `Close()` on each endpoint, each under its own
`try` so one SDK failure cannot strand the endpoints behind it, and swap the
collection out under the lock first so a second call closes nothing twice. A
writer starts no reader thread, so a dropped publisher did reach `__del__`; the
change makes the release happen at the call rather than whenever the last
reference happens to go.
