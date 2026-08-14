### Quality: pin `PolicyRunner.evaluate`'s per-episode reset tolerance and its async-RTC success detection

`evaluate` documents that a failed `policy.reset(seed=...)` warns and continues,
and its async-RTC branch carries a comment saying its success check "mirrors the
synchronous path / `_evaluate_with_spec`". Both arms were unexecuted, while the
equivalents in `_evaluate_with_spec` are covered - so the two parity claims held
only by inspection. Ten cases now drive them: a raising reset still evaluates
every episode and records one warning naming the seed, the reported success rate
is unaffected, a `CooperativeStop` from reset still propagates, and the async
path detects success against the live post-action observation at the same step
the synchronous path does. No library behaviour changes.
