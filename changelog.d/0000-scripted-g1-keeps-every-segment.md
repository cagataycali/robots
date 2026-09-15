### Fixed: `scripted_g1.py` delivers every segment it records, and `concat_clips` joins recorded clips

`examples/locomotion/scripted_g1.py` drove its four-segment locomotion
schedule as four `run_policy` calls that all named the same `--mp4`, saying it
would "append each segment to one MP4". A rollout opens its video fresh, so
the file held only the last segment - the two-second halt - and the example's
"reproducible demo artifact" was a video of a robot standing still.

`strands_robots.rendering.concat_clips(paths, out, fps=None)` joins clips end
to end through `encode_clip`, reading the rate from the first clip's header
unless one is passed, and refuses an empty list, a missing clip, a segment of
another frame size, or a clip that declares no rate - each naming the clip. A
GIF header declares a per-frame delay rather than a rate, and that delay is
what `encode_clip` writes for a GIF, so the rate is read back through the
inverse of the same conversion: the `encode_clip` -> `concat_clips` round trip
closes for both containers `encode_clip` writes, not just MP4.
The example records each segment to `<stem>.seg<i>.mp4`, joins them into
`--mp4`, removes the segments unless `--keep-segments`, and says why. It also
mounts a camera on the pelvis (`add_camera(parent_body="unitree_g1/pelvis")`)
before the first rollout and records every segment from it: the scene's
`default` camera frames the origin from a fixed vantage and loses a walking G1
within a couple of seconds, so the demo it recorded was mostly empty floor.
