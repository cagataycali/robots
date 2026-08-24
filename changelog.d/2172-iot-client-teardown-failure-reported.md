### Fixed: `IotMqttTransport` records every MQTT5 client teardown it could not finish

`self._client.stop()` is called from four places in the AWS IoT transport, each
tearing down a client the transport is about to stop referencing. Two of them
contained a failing teardown and logged it at debug; the construction-failure
one states the policy in a comment ("a stop() error here ... must not mask the
original failure. Log at debug and move on."). The other two did not follow it.

The connect-timeout path called `stop()` unwrapped, so a raising teardown left
`connect()` - documented to return `False` when the broker is unreachable within
`connect_timeout` - raising `RuntimeError` instead, with `self._client` still set
and its IO thread still running. `close()` swallowed the same failure into a bare
`pass` and then logged "IoT mesh session closed", the only line an operator gets
and the opposite of what happened; because `close()` drops the client reference
either way, nothing can reach that client afterwards to retry.

Both now tolerate the failure and record it - the timeout path at debug beside
the two paths it mirrors, `close()` at warning because it is the only teardown
whose visible report is otherwise a success. A structural test pins the rule
across all four sites so a fifth cannot ship without it.
