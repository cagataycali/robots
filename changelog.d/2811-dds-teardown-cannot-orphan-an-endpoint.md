### Fixed: a DDS endpoint built across a teardown is released, not left live

`DDSSubscriberSet.subscribe` and `DDSPublisher.get_publisher` build their SDK
endpoint under the shared construction lane `_DDS_INIT_LOCK`, because the
CycloneDDS bindings segfault on concurrent construction. `close` swaps its
collection out under the instance `_lock` and releases what it took. Those two
locks do not exclude each other, so a teardown could complete between the
moment a construction finished and the moment it recorded what it built, and
the recording then landed in a fresh list or dict that `close` had already
walked past.

Nothing reached the endpoint afterwards. `G1Driver.cleanup` drops the set right
after closing it, so the second `close` that would have collected the orphan
never happens - and `cleanup` does not read the outcome of the `loop.stop` that
precedes it, so a control loop whose join times out keeps calling `publish`,
rebuilding a publisher into the emptied cache on the way. What was left behind
is the state the engine's own `close` docstring describes at length: `subscribe`
asks for `queueLen=10`, and at any queue length above zero the SDK starts a
`ch_reader` daemon thread that keeps the channel reachable, so the reader stays
matched and the decoder callback keeps filling caches for a driver that believes
it is disconnected.

Each recording path now captures the collection it was told to write to and
compares it before recording, releasing through `_release_partial` and naming
the topic when a teardown moved it. `_release_partial` already existed to
uphold this invariant where construction *fails*; the same rule now covers the
path where construction succeeds and the recording is what could not happen.
Sequential use is unchanged: a `subscribe` or `get_publisher` after a `close`
captures the collection that `close` left behind, so a driver reconnecting is
never refused for a teardown that already finished.
