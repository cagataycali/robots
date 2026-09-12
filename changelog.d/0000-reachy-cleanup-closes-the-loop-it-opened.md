### Fixed: the Reachy driver closes the asyncio loop its link ran on

`ReachyDriver` runs its real-time link on a background asyncio loop from
`asyncio.new_event_loop`, whose documented counterpart is `loop.close()`.
Teardown called `loop.stop()` instead, which only asks `run_forever` to return
and leaves the loop's selector and self-pipe open. So every connect/teardown
cycle abandoned one open loop - on the success path through `cleanup()` and on
both give-up paths through `_release_link`, where a retried bring-up abandoned
one per attempt - and Python raised a `ResourceWarning: unclosed event loop` for
each, reported wherever the collector happened to reclaim it rather than at the
teardown responsible.

Closing it means waiting for the loop's thread first: `stop()` is asynchronous,
so the thread is still inside `run_forever` when it returns, and closing a
running loop raises `RuntimeError`. Teardown now waits for that thread, bounded
by `_LOOP_JOIN_TIMEOUT_S` so a link callback wedged on a socket read cannot hold
teardown open for as long as the read takes, and then closes the loop. This is
what `_loop_thread` was kept for; it had no reader before. A thread that
outlasts the budget keeps its loop - it is by definition still running, so
closing it would raise - and that outcome is logged rather than passed off as a
teardown that finished.
