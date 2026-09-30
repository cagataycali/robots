### Fixed: `cosmos3` declares the chunk it serves, so a rollout keeps the whole chunk and can mask its latency

`Cosmos3Policy` returned 16- or 32-step chunks but declared
`execution_horizon == 1` and `is_chunk_emitting() == False`, so the runner
dropped every chunk's tail at the default `action_horizon` and
`run_policy(async_rtc=None)` never enabled async RTC for it. The constructor
now declares `actions_per_step` (the embodiment's `action_chunk_size`, or a
value you pin), and the served chunk length replaces the default the first
time the server answers unless you pinned one. `MockPolicy` keeps declaring
`1` and the base class now says why. (#4189)
