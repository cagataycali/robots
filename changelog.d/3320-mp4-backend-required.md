### Fixed: an MP4 request without an MP4 backend is refused as the missing dependency

Every writer here probed `imageio` before writing, and `imageio` is two
dependencies: the package, and the plugin that encodes the container. `imageio`
declares that plugin - `imageio_ffmpeg` - as an optional extra of its own, so an
install can supply the module the encoders import and still have no MP4 writer
behind it. The `[vera-sim]` extra is such an install (it declares `imageio`
alone), and its own MimicGen example records with an `.mp4` default.

With the plugin absent `imageio` did not refuse the `.mp4` request. It fell
through to whatever other plugin claimed the container, and the libx264 knobs
reached a writer that had never heard of them, so the request failed as
`PyAVPlugin.write() got an unexpected keyword argument 'quality'` - a `TypeError`
naming a writer the caller never asked for, which is neither what the encoder
documents nor what its callers handle. Measured at all three writers:

- `stop_cameras_recording` (MuJoCo), on a real recording holding 8 buffered
  frames: `status="success"` with 0 frames written, a 0-byte `clip__cam_a.mp4`
  left on disk and the recording deregistered - all 8 frames lost under a
  success verdict. The `except ImportError` branch that keeps the recording
  registered for a retry was unreachable for this condition.
- `stop_cameras_recording` (Isaac) catches only `ImportError`, `RuntimeError`
  and `ValueError`, so there the `TypeError` escaped a method documented never
  to raise, after the recording state had already been cleared.
- `run_policy(video=...)`: the rollout ran to completion and reported
  `Policy failed: PyAVPlugin.write() got an unexpected keyword argument
  'quality'` beside a 0-byte MP4.

The container now decides which modules must be present, and one function
decides it - `require_clip_encoder` (`strands_robots.rendering`), which every
writer routes its probe through instead of repeating the rule: `.gif` needs no
plugin (Pillow writes it, and `imageio` hard-requires Pillow), anything else is
MP4 and requires `imageio_ffmpeg`. The refusal comes from `require_optional`, so
it names the module actually absent and `pip install
'strands-robots[sim-mujoco]'` - the extra whose bounds this project owns -
rather than two bare distributions. It is raised before the frames are read, so
a caller keeps its iterable and can install and retry.

Consequently the same MuJoCo recording now reports `status="error"` with the
recording still registered holding all 8 frames, and a second stop with the
plugin installed encodes them; the rollout returns
`_RolloutVideoWriter.open`'s error envelope - like every other setup failure
there - before the loop runs, writing nothing. Both camera flushes quote that
refusal instead of their own fixed `"imageio not installed. pip install imageio
imageio-ffmpeg"` line, which named the wrong module whenever it was the plugin
that was missing.
