### Tests: the shared VERA docker query bound is held to a span a healthy daemon can meet

`_DOCKER_QUERY_TIMEOUT` is what lets `DockerServerRunner`'s readiness wait reach
its own deadline instead of merely writing it. That the queries state a bound, and
that a daemon which does not answer inside it is reported rather than answered
for, is graded. Two properties of the bound itself were not, both measured by
mutating `server_runner.py` and running the existing file alone.

A bound of `0.0` is an over-correction rather than a return to the old state:
`subprocess.run(timeout=0.0)` raises `TimeoutExpired` before the child can answer,
so every query fails against a perfectly healthy daemon and `server_mode="docker"`
stops working altogether - and the twelve existing cells all still passed. It is
invisible where the daemon is a callable standing in for `subprocess.run`, because
no bound is ever applied to a real child; the control added here invokes a real
executable that answers promptly, for both the running and the not-running
verdict, which is what makes "bounded" distinguishable from "refuses everything".
`None` is the other end of the same axis.

The log read reverting to its own literal `10` also left all twelve passing,
because `10 == 10.0` satisfies a cell comparing the value. The constant's comment
describes the probe and the log read as sharing one bound, so a cell now holds
them to it and a future change to the bound cannot leave one behind.

`docker run` is still not a query and still states no bound, and the SIGTERM grace
period `stop` waits out keeps its own larger one; neither is in scope.
