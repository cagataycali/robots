### Fixed: `evaluate_benchmark` reports inference cost and stops querying the policy once the horizon is spent

The benchmark result now carries the `avg_inference_ms` / `max_inference_ms`
pair and the chunk counters `eval_policy` already reported, collected through
the same seam, so the docs' promise holds for both eval surfaces. Its episode
loop is bounded by the steps applied instead of running one iteration per step:
with `action_horizon=8` it queried the policy eight times per chunk it used,
paying that many VLA inferences for nothing. Closes #4146.
