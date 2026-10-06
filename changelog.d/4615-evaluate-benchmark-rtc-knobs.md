### Fixed: `evaluate_benchmark` takes `async_rtc` and `rtc_inference_timeout_s` like its siblings

`SimEngine.evaluate_benchmark(async_rtc=True)` raised a bare `TypeError`, while
`run_policy`, `start_policy` and `eval_policy` all take both RTC knobs. It now
accepts them: `async_rtc=True` is refused before any policy is built with the
reason (a benchmark stays synchronous so its success rate is bit-stable) and the
sibling to use instead (`run_policy(async_rtc=...)`); a non-boolean flag or an
out-of-domain deadline gets the same refusal `eval_policy` gives.
