### Fixed: `lerobot_local` refuses an unknown `embodiment=` before any weights download

`embodiment="so102"` was resolved by `_configure_embodiment`, which runs after
`_load_model`, so `run_policy` downloaded and loaded the checkpoint and only
then raised `RuntimeError: Failed to load embodiment 'so102'` (18 s cold),
while a misspelt keyword on the same call was an envelope in 0.09 s. One rule,
`embodiment_spec_error`, is now read by `preflight` (so `run_policy`,
`eval_policy`, `evaluate_benchmark` and the physical arm answer `status=error`
naming the registered embodiments before the download) and by the constructor
(so a direct `create_policy` caller is refused before `_load_model`). (#4163)
