### Fixed: `run_policy(policy_provider="remote")` reports an unusable remote instead of raising past its envelope

`websockets` is not a base dependency, so a base install could resolve a release
older than 17.1 through another package, and the first rollout step then raised
`TypeError: create_connection() got an unexpected keyword argument 'legacy'`.
`RemotePolicy` and `Cosmos3WebsocketClient` now refuse at construction with the
missing-dependency `ImportError` other optional providers give, naming the
installed release and `pip install 'strands-robots[inference]'` (or
`[cosmos3-service]`). Separately, a remote server nobody listens on raised
`ConnectionError` out of `run_policy`, `eval_policy`, `evaluate_benchmark` and
`run_multi_policy`, because the client's first contact is the rollout's
`requires_images` read, made before the loop that reports policy failures. Each
surface now returns `status="error"` with the client's own "could not reach a
PolicyServer" report and `stopped_reason="error"`.
