### Fixed: every rollout loop acquires action chunks through one pipeline and one query helper

`PolicyRunner` carried three hand-rolled copies of the chunk-query body and two
copies of the async-RTC prefetch pipeline: `run()` re-implemented inline what
`_ChunkPipeline` already provided for `evaluate()`, complete with its own
executor, its own `prefetch_trigger`, its own `_swap_in` and its own starvation
warning. The copies had drifted, so the same degenerate policy produced four
different outcomes depending on which entry point drove it - and the worst of
them was unbounded.

`run(async_rtc=False)`, the default rollout path, had no empty-chunk guard at
all. A policy returning `[]` never advanced `step_count`, so the loop re-queried
forever: a 4-step budget was measured issuing 112,200+ `get_actions` calls in 25
seconds and still climbing, with no video, no error and no way out but a
timeout. The pipeline's synchronous iterator - which `evaluate()` already used -
refuses on the first empty chunk in one line, and `run()`'s own async branch
refused too. Routing both `run()` paths through the pipeline makes the default
path refuse after exactly one query.

`run()`'s synchronous branch also never counted the chunks it acquired, so the
documented `rtc_chunks_acquired` field reported `0` beside a non-zero
`rtc_avg_inference_ms` - telemetry that contradicted itself, while `evaluate()`
reported `3` for the identical policy and horizon. The counters now live on the
pipeline instance and are read back for both the success and the error payload,
so a rollout that dies mid-flight still reports the chunks it did acquire.

`_evaluate_with_spec` was the one rollout path that never declared the RTC
chunk-seam offset. `rtc_observed_delay_steps` is persistent policy state, so a
policy handed over from an async rollout kept slicing its chunk against that
stale count on a loop where the world advances 0 steps during inference: a
policy left at `7` was measured seeing `7` on all six steps of a synchronous
benchmark episode, and now sees `0`. A policy driven by `evaluate_benchmark`
must therefore implement `Policy.set_rtc_observed_delay`, which every `Policy`
subclass inherits concretely; `run()` and `evaluate()` already required it.

The two prefetch formulas that had drifted textually
(`len(cur_chunk) - prefetch_trigger` against `len(cur_chunk) - idx`, equal only
because both fire on the step the trigger fires) collapse to the `idx` form,
which is the remaining-step count by construction. `_ChunkPipeline` now also
publishes `chunk_index` and `observation_age_steps` per yield, keeping the two
facts a recording consumer needs distinct: the first action of a *prefetched*
chunk is not a reused observation, yet its observation is already several steps
old. The budget is still tested before a chunk is pulled, so no rollout pays an
extra inference for an action it can never apply.
