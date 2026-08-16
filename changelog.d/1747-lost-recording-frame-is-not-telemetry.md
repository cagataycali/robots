### Fixed: a dataset frame the recorder could not write now fails the rollout

`DatasetRecorder` defaults to `strict=True` and raises when a write fails, but that
raise reached the rollout drivers through the same `on_frame` hook they use for caller
telemetry, which tolerates a bounded number of *consecutive* failures and resets its
counter on every success. An intermittent write failure therefore never reached the
limit: a 20-step rollout that lost every other frame reported `status="success"` and
`stop_recording` reported a saved episode, while the dataset held 10 frames re-stamped
from the declared `fps` into half the span they were captured over.

The recorder now raises `RecordingFrameError` (chaining the original failure), and
every rollout loop excludes that type from the telemetry tolerance - the rollout ends
at the first lost frame with an error naming it. A caller's own `on_frame` hook keeps
its tolerance, and `strict=False` recording still drops, counts and continues.
