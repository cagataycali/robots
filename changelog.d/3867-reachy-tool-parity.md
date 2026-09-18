### Added: Reachy Mini - the whole vocabulary on the native driver, out of the box

`Robot("reachy_mini", mode="real")` needs no port, no `driver=` and no connect
line: the daemon is discovered (`$REACHY_HOST`, then `localhost`, then
`reachy-mini.local`, skipping a daemon that reports its own start-up error), and
the first verb that needs it connects. `status` never connects, because it is the
question "are we connected". `mode="auto"` asks the driver's `probe_hardware()`
before falling back to sim.

The tool spec grows from 6 verbs to 24, one handler per verb in a single dispatch
table the schema enum is generated from. Motion: `look` (smooth minjerk head
pose in degrees and millimetres, optional body yaw and antennas), `antennas`,
`body_turn`, `home`, `wake`, `sleep`, `express`, `list_moves`, `motors` and
`look_at`, which turns the head toward a camera pixel resolved through the
robot's own calibration. `express` takes plain words - `happy`, `curious`, `no` -
as well as library names, from the emotions and dances libraries, and the
path-segment gate runs before any request. Media: `say` through a TTS sidecar
(`tts_url=` or `REACHY_TTS_URL`, refused by name when absent), `play_sound`,
`volume`, `set_volume` - which requires `allow_test_sound=true`, because the
daemon plays a short test sound on every level change - `camera` and
`record_audio`. Attention: `track_face`, `tracking_status`, and `turn_to_sound`
with `turn_to_sound_status`, a 10 Hz poller that makes one smooth turn per
utterance and is vetoed by a fresh face-tracker lock or a move in flight.

`stop` enumerates the daemon's running moves and stops each by uuid. The
daemon's stop endpoint takes one uuid, so the bare POST it replaces was a 422 and
halted nothing.

Every write returns the daemon's acceptance and says so with
`motion_verified=false`: acceptance is not proof that the head reached the pose
or that audio was heard. The daemon also accepts a move in every torque mode, so
`motors` with no `mode` reports the mode the robot is actually in - a move
commanded with torque off answers with a move uuid and moves nothing.
