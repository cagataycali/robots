The Device Connect caller allowlist advisory now follows the transport that
carries the call rather than one of the two sources that decide it. A
`DeviceRuntime` resolves its posture from its own `allow_insecure` argument first
and `DEVICE_CONNECT_ALLOW_INSECURE` second; the advisory read the variable alone,
so a device brought up with `allow_insecure=True` and the variable unset ran
insecure with a configured allowlist and logged nothing - the shape the package
guide documents for `ReachyMiniDriver` - while one brought up with
`allow_insecure=False` and the variable opting in ran authenticated and was
warned about anyway, the message citing the variable as the reason.

`is_authorized_caller` now takes the `DeviceRuntime` the driver is attached to
and reads its resolved `allow_insecure`, falling back to the variable only when
no runtime is attached. The variable's opt-in vocabulary was spelled once per
reader in three modules and is now spelled once in the stdlib-only `_authz`,
which the resolver and the agent-side connector import; a derived guard holds any
further reader of the variable to that.
