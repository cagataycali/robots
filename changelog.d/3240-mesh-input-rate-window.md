### Fixed

- **A restarted mesh teleop stream no longer reports a rate it never ran at.**
  Both sides of the input stream report an achieved rate beside a frame total -
  `InputPublisher.stats` as `{"frames": N, "hz_actual": ..., "hz_target": hz}`,
  `InputReceiver.stats` as `{"frames_received": N, "hz_actual": ...}` - and
  `get_teleop_status()` prints both lines from those keys, so one reading is
  compared against one `hz_target`. `hz_actual` was `frame_count / elapsed`, and
  its two operands were measured over different windows: the counter is
  cumulative for the life of the object (both `stats` docstrings say so, and the
  receiver's sampled safety audit keys its cadence on that count) while
  `_start_mono` is re-stamped by every `start()`.

  Restart is a supported flow, not a corner: the publisher's `start()` clears
  the stop event its `stop()` set, the receiver's re-declares the subscription
  `Mesh.stop` drops ("a caller that rejoins the mesh re-declares its own
  subscriptions"), and `InputPublisher.stop` documents retrying a join that timed
  out. Measured over a real 30 Hz stream between two Zenoh peers, each driving a
  MuJoCo SO-101 arm: session one reported `30.07` Hz, and after a stop and a
  restart **every one of the 24 status polls during session two reported a rate
  above the 30 Hz the stream was running at** - `994,648` Hz at the first poll
  (a full session's frames divided by a window microseconds old), `115.74` Hz two
  seconds in, and still `60.44` Hz six seconds in. A rate above `hz_target` is
  the one reading no averaging of an achieved rate can produce, and `hz_actual`
  against `hz_target` is the documented way to judge a teleop link, so a
  degraded link after a rejoin read as a healthy one.

  `start()` now records the frame count beside the clock stamp, and the
  arithmetic has one owner (`_achieved_hz`) because both sides report it under
  the same key. The numerator covers the window the denominator does; every
  reported total stays cumulative, so the receiver's audit cadence and the
  `errors == frames_received` signature a fully-refusing follower shows are
  unchanged. The same run reported `29.44` Hz two seconds into the restarted
  session, with one of 24 polls `1.97` Hz over target - ordinary jitter within a
  quarter-second sampling window.
