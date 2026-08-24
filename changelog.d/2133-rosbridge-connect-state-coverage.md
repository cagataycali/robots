### Quality: pin the two `use_rosbridge` connection states the roslibpy double could not express

`_RosbridgeBackend.connect` branches on six states; four were pinned. The two that were not are
the recovery states a real bridge produces, and one of them looked covered: the test named for the
auto-reconnecting factory installed a *non-data* descriptor over `is_connected`, which an instance
attribute set by `run()` silently shadows, so the scripted reads were never consulted and the test
returned from the plain cache-hit branch instead of the wait loop it names.

The double can now express a dial that reports ready while the connector reads as disconnected -
`run()` waits on a one-shot `on_ready` event, while `is_connected` live-reads
`connector.state == "connected"`, so returning from the first does not imply the second. The
descriptor gained `__set__` so the scripted reads decide, plus an assertion that the loop really
polled. `use_rosbridge.py` goes from 97.7% to 98.85% covered, with `connect`'s state matrix at 6 of 6.
