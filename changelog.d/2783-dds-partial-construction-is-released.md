### Fixed: a DDS endpoint whose construction failed part way is released, not dropped

`DDSSubscriberSet.subscribe` and `DDSPublisher.get_publisher` build an SDK
endpoint in two steps - construct, then `Init` - and report a named reason if
either raises. Neither step is atomic, and the endpoint that failed holds real
DDS state: `ChannelSubscriber.__init__` creates a live `Channel` before `Init`
runs, and `unitree_sdk2py` starts the `ch_reader` daemon thread *before* it
constructs the `DataReader` that can raise. Both handlers returned the reason
and dropped the endpoint.

That leaks exactly what the engine's `close` methods exist to release, and it
leaks it where `close` can never reach: the failing call never appended to
`_subs` or wrote to `_pubs`, so `G1Driver.cleanup()` walks a collection the
half-built endpoint is not in. Measured against `unitree_sdk2py` 1.0.1 by
driving `subscribe` with a `DataReader` that raises:

```
subscribe() reason        failed to subscribe to 'rt/lowstate': DDS resource limit reached
subscribers recorded      0
ch_reader threads live    1     <- before this change
after close()             1     <- close() has nothing to walk
ch_reader threads live    0     <- after this change
```

The thread runs for the life of the process, blocked on its queue, keeping the
channel and its `DataReader` reachable so no finaliser releases them - the same
reference chain `DDSSubscriberSet.close`'s docstring already describes when it
explains why dropping a reference is not a release. A driver that retries
`connect_eagerly()` after a transient DDS failure accumulates one more per
attempt, four topics at a time.

Both construction sites now pass the half-built endpoint through one release
helper before returning. `Close()` is safe on a partial init - it routes to
`Channel.CloseReader`/`CloseWriter`, which skip an entity that was never created
and still stop and join the reader thread - and a `Close()` that fails is
reported rather than allowed to replace the construction failure the caller
needs. A constructor that raised before there was an endpoint releases nothing
and says nothing about closing one. The reason strings both methods return are
unchanged, and a construction that succeeded is still recorded, still cached and
still closed only by `close`.

The write path gets the same rule for the reason its own `close` docstring
gives: a dropped writer starts no thread and does reach CycloneDDS' finaliser,
but only "whenever the last reference happens to go", and closing "says when".
