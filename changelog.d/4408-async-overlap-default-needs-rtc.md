### Fixed: `run_policy` keeps a chunk policy without RTC synchronous by default

`async_rtc=None` used to switch on the background chunk prefetch for every
chunk-emitting policy. The prefetched chunk is planned from an observation half
a chunk old, and a policy without Real-Time Chunking swaps it in unblended, so
its first actions target a state the robot has already left (ACT,
diffusion, MolmoAct2 and `lerobot/smolvla_base`). The default now overlaps only
a policy that also reports `supports_rtc`; every other policy runs the
synchronous chunk-then-drain loop. Pass `async_rtc=True` to keep the old
latency masking for a non-RTC chunk policy.
