### Tests: the unthrottled control-loop pin measures the throttle, not the runner

`TestPeriodIsTheOnlyThrottle` asserted that a free-running control loop applies
more than 1000 actions in a 0.2 s window. That count is a property of the host
and of the coverage tracer the suite runs under, not of the throttle being
tested: on `ubuntu-latest` runners it lands at 570-693 under `--cov`, so `main`
went red on every push once the pin merged, and because the suite runs under
`-x` roughly 17% of the tests stopped executing with it.

The bound is now relative to the throttled loop measured in the same process,
which `duration * control_frequency` caps at the same value on any host. The
`control_frequency` guard the test covers is unchanged.
