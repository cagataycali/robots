### Fixed: `MockPolicy.reset` rewinds the sinusoid; `PersistentPolicy` reports the clock it forwarded

The mock's `_step` counter advanced across episodes with nothing to rewind it,
so two episodes seeded alike began at different phases - and every wrapper
forwarding `reset` to a mock (composite, persistent, remote over a mock
server) inherited the drift. `reset` now sets the counter to zero. Separately,
`PersistentPolicy.control_frequency` and `rtc_observed_delay_steps` read `None`
after their setters ran, because both are class attributes on `Policy` and the
wrapper's `__getattr__` never saw the read; they are now properties over the
wrapped policy's values (#4191).
