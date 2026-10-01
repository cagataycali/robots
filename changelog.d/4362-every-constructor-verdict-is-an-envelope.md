### Fixed: a policy constructor's verdict is a `status=error` envelope on every rollout surface

`run_policy`, `eval_policy` and `evaluate_benchmark` turned a constructor's
`TypeError` or `ValueError` into their documented `status=error` envelope and
let every other exception escape as a raise: a `lerobot_local` checkpoint id
that is not on the Hub (`FileNotFoundError`), a `remote` server nobody listens
on (`ConnectionError`), an `rl` checkpoint directory without its
`policy_meta.json` and a `wbc` directory without its ONNX (`RuntimeError`).
The boundary is now one shared rule,
`strands_robots.policies.construction_failure_keeps_its_raise`: the
remote-code gate and a missing optional dependency (an `ImportError`, or a
provider error wrapping one as its cause) keep their raise, every other
constructor raise is this configuration's verdict and travels in the envelope,
followed by the configuration judged so the checkpoint id or address is named.
(#4153)
