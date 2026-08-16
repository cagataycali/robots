### Fixed: opening a recording against a running rollout no longer mislabels the episode

A dataset's declared `fps` must be the rate its frames were captured at: the
recorder is driven once per control step with no decimation, and LeRobot derives
every timestamp from `fps` positionally. Every rollout entry point already
refused a `control_frequency` that disagreed with an open recording, but the
inverse ordering was unguarded. `start_policy` submits its rollout and returns
while it continues, so a recording could be opened against a rollout already in
flight - and the two library defaults collide (`fps=30` against
`control_frequency=50.0`), so no unusual argument was needed:

```python
sim.start_policy(robot_name="arm", policy_object=policy)   # 50.0 Hz
sim.start_recording(repo_id="local/ds", task="t", fps=30)  # was: success
...
sim.stop_recording()   # was: success, "81 frames, 1 episode(s)"
```

That episode declared 0.0333 s between frames captured 0.0200 s apart: a 2.6667 s
episode for a 1.62 s capture, with all three calls reporting `status="success"`
and no log line. The distortion is the control period a policy trains on, and
`replay_episode` derives its per-frame physics budget from the dataset rate, so
the same episode also replays at the wrong speed.

`start_recording` now refuses the disagreement before creating the dataset,
naming the running rollout, both rates and both remedies - record at the
rollout's rate, or stop the rollout and restart it at the recording's rate. The
explanatory text is shared with the rollout-entry guard, so the two orderings
cannot describe the same distortion differently. Rollouts running at *different*
rates are refused outright even when `fps` matches one of them: their frames
interleave into one episode whose single declared rate cannot describe both, so
there is no `fps` to pass instead.

Matching rates are unaffected, and a backend with no asynchronous rollout is
unaffected: the new `SimEngine._active_rollout_rates()` hook reports an empty
mapping unless a backend overrides it, so the guard is inert where no rollout can
be in flight when `start_recording` is reached.
