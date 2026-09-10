### Fixed: the RTC prefix names the actions the robot has not executed yet

LeRobot fixes what `prev_chunk_left_over` means in
`ActionQueue.get_left_over()`, which its inference loop snapshots immediately
before denoising: `original_queue[last_index:]`, the previous chunk from the
observation tick to its end. Row 0 is therefore the action applied on the tick
right after the observation, and row *i* the action applied on the tick the new
chunk's row *i* lands on. That index alignment is the mechanism -
`RTCProcessor.denoise_step` builds
`get_prefix_weights(inference_delay, execution_horizon, T)`, which pins weight
1.0 across `[0, inference_delay)` (the steps that elapse *during* inference) and
blends `[inference_delay, execution_horizon)` toward the prefix.

`LerobotLocalPolicy` cut the prefix at emit time instead, keeping the chunk tail
past the execution horizon. Which part of a chunk becomes the next prefix is not
knowable then: it depends on how far the consumer has drained the chunk when the
next observation is captured, and that count only arrives with the next
inference (`set_rtc_observed_delay`). On the synchronous path the guess happens
to be right, because the chunk is fully drained before the re-query. On the
async rollout path the runner fires the prefetch when the chunk is half drained,
so the prefix was shifted forward by exactly `observed_delay` steps on every
seam: the denoiser froze the new chunk onto actions the robot would not reach
for another `observed_delay` ticks, and blended the seam against actions the
runner discards at the swap and never executes.

The policy now keeps the chunk as the consumer received it - LeRobot's
`original_queue` - and derives the prefix at the next inference from the overlap
the runtime reports, which is that queue's `last_index` seen from the other end.
Measured on the real async pipeline with `execution_horizon=10` and a 50-step
chunk: the prefix opened on the action 5 ticks ahead of the one being executed,
and lerobot's own weight schedule hard-froze all five of those rows. The
synchronous path, where nothing is pending during inference, is unchanged.
