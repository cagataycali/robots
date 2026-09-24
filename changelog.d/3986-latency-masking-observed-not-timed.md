### Fixed: latency masking is observed, not timed

The two async-RTC cells that pinned latency masking ran the loop twice and
compared the wall clock (`sync_elapsed - async_elapsed > 2 * _INFER_SLEEP`). The
saving being measured is 0.1s of a ~0.5s run, so a loaded machine perturbs the
subtraction by more than the margin: under CPU contention the pair went red in 2
of 6 runs here, missing by 2ms (`0.5381 - 0.4403`), and the cost lands on the one
required check, which does not distinguish a flake from a regression.

Each now holds the query open until the loop dispatches another action and
asserts the overlap it is named for: the async pipeline drains the rest of the
current chunk while the prefetched query is in flight, while the synchronous loop
is inside the call and cannot dispatch, so it falls through on the timeout. Both
modes are one table, so the contrast is the pin.
`tests/simulation/test_policy_runner_async_rtc.py` already carried this remedy
for its sibling overlap cell ("proven by a rendezvous, NOT a wall-clock timing
race"); these two were the cells left behind.
