### Bug Fixes

- **tools/lerobot_camera**: `REALSENSE_AVAILABLE` now reports whether a
  RealSense camera can be *opened*, not whether lerobot's camera classes
  import. lerobot imports `lerobot.cameras.realsense.*` whether or not the
  Intel SDK is installed - `pyrealsense2` is bound to `None` behind an
  availability flag and required at the call sites instead - so the flag was
  `True` on every host with lerobot and no SDK. `camera_details` then answered
  `SDK Available:  Yes` / `Depth Support:  Yes` with `status="success"` for a
  host that cannot open one, the branch that reports the SDK absent and names
  the install became unreachable, and camera discovery called into lerobot only
  to log `'NoneType' object has no attribute 'context'` as a discovery failure.
  The same session's `capture` disagreed, because lerobot's own `require_package`
  raised there naming the install. The flag is now derived from the SDK under
  its import name - the one both the `pyrealsense2` and `pyrealsense2-macosx`
  distributions provide - and both surfaces report an absent SDK from one owner,
  `REALSENSE_SDK_ABSENT`. That message names lerobot's `intelrealsense` extra
  rather than the `pyrealsense2` distribution, because the extra carries the
  per-platform split the bare install misses on macOS. A `realsense` request
  with no SDK is refused as the absent SDK (an `ImportError` carrying
  `name="pyrealsense2"`) instead of as `Unsupported camera type: realsense`,
  which sent the caller looking for a spelling that does not exist; an
  unsupported type keeps that refusal.
