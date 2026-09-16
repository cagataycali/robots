### Fixed: `add_robot` refuses while a dataset recording is active

The recorder freezes its feature schema from the robots present at
`start_recording`; a robot added afterwards had no columns, so the next
rollout failed inside lerobot with a feature mismatch and left an empty
dataset shell on disk. `add_robot` now refuses during an active recording and
names the sequence (stop, add, start again). A robot-less `start_recording`
stays allowed as a camera-only recording, and its success text now says so.
