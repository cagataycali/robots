### Tests: twelve cells and one fixture wait for the event they assert, not a production timeout

Three MuJoCo recorder-warmup cells sat out `start_cameras_recording`'s whole
warmup budget for a thread they never start; two dashboard cells parked the
render the create route waits on, or the one a dropped session's stop joins;
the teleop join fixture left a loop polling after a cell that never stopped it,
and timed out joining it; seven mesh sensor-loop cells slept up to 5.5 s for a
first publish that lands in milliseconds. Each now waits for what it asserts.
Same covered lines; the four files run in 40 s instead of 84 s.
