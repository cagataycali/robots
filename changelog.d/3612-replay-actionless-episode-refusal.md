### Fixed: `replay` refuses an episode whose frames carry no recorded action

`PolicyRunner.replay` / `SimEngine.replay_episode` tolerate a single frame with
no recorded `action` by advancing physics for its control period, so the frame
still occupies its recorded time slice. That frame was also counted as APPLIED,
which made the degenerate case invisible: an episode where NO frame carried an
action - an observation-only dataset, or one whose actions live under another
column name - returned `status="success"` with `Frames: 40/40`, text
byte-identical to a replay that reproduced the trajectory, while `send_action`
was never called and the arm only sagged under gravity (0.03 rad of motion
against 0.49 rad for the same episode replayed with its action column).

That contradicted the documented contract in both spellings and in
`docs/recording.md`: a success status means the recorded frames reached the
actuators. A dataset's column schema is fixed for the whole episode, so this is
a property of the dataset rather than of one frame, and the refusal names the
columns the frames DO carry - the signal for the near-miss column name behind
it. The status `json` block now reports `frames_with_action` beside
`frames_applied`, so a partial per-frame gap is read from the two counts instead
of being hidden behind one number.
