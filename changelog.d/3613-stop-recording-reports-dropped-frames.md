### Fixed: `stop_recording` reports the frames a `strict=False` recorder dropped

A `strict=False` `DatasetRecorder` drops a failed dataset write and counts it in
`dropped_frame_count` instead of raising, so the rollout still reports success.
`stop_recording` is where that count has to be read, because it releases the
recorder as it returns - a count it does not report is a loss nothing can
measure afterwards, and it reported none.

Both shapes of loss were invisible. With every write failing, the dataset was
empty and the empty-dataset refusal blamed the rollout loop - telling a caller
who had just run `start_recording` -> `run_policy` -> `stop_recording` that
frames come from `run_policy` and to run that recipe, the one cause that cannot
apply when `run_policy` had fed the recorder 20 frames. Measured: 20 attempted,
0 written, 20 dropped, refused with a message naming none of it. That case now
names the failed writes and the `strict=True` remedy. With half the writes
failing, the session stayed a `status="success"` reading `10 frames, 1
episode(s)` - identical to a 10-frame session that lost nothing. A partial loss
is still a success, because `strict=False` documents dropping a failed write and
completing, but it now says how short it is in the text and in a
`dropped_frame_count` field beside `frame_count` in the status `json`.
