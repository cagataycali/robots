### Fixed: a GR00T server that does not answer is reported fast, with the remedy

`run_policy(policy_provider="groot")` against a port nothing listens on hung
15 s and then reported `Policy failed: Resource temporarily unavailable` -
`zmq.Again` escaping the client raw, with no host, port, budget or remedy -
and `policy_config={"timeout_ms": 500}` did not shorten it, because
`Gr00tPolicy` took no `timeout_ms`.

`Gr00tInferenceClient.call_endpoint` now raises `ConnectionError` naming the
`tcp://` URI, the endpoint, the `timeout_ms` that expired and, from a 1 s TCP
probe, whether anything is listening there: connection refused gets the
start-a-server remedy (`gr00t_inference(action='start', port=N)` or
`python -m gr00t.eval.run_gr00t_server --port N`); a listening but silent
port gets "still loading or wedged: read its log, raise `timeout_ms`". The
REQ socket is re-created after a timeout, so a retry is a real attempt.
`Gr00tPolicy` gains `timeout_ms` (default 15000, unchanged) and forwards it.
