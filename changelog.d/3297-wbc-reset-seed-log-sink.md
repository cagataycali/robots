### Fixed: the seed a `WBCPolicy.reset` renders into its log line is escaped at the sink

`WBCPolicy.reset` rendered the `seed` it was handed into its debug record with
`%r`. `PolicyServer` forwards the wire `seed` to `reset` verbatim - the policy
owns the domain - so the value in that record is caller-side input, and CodeQL
reported the sink as `py/log-injection` (alert 1159). A JSON-decoded value
cannot carry a raw break through `repr`, so the escape was incidental to the
wire's encoding rather than a decision at the sink - the same shape #2853
measured at the two `%r` joint-state sinks. The rendered value now passes
through `sanitize_log_value`, the one owner of that escape, and the sink is
named in the census `tests/policies/test_log_sink_sanitizer.py` grades, beside
the eight it joins. Pinned there: a seed whose `repr` spans two lines no longer
splits the record.
