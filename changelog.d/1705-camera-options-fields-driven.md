### Fixed: a per-camera option that cannot be honored is refused, and every declared field is reachable

`Robot(..., cameras={"front": {...}})` read each camera's dict key by key
against a hand-picked list of six options. Two consequences, both silent:

- `warmup_s` and `backend` were **unreachable**. lerobot's
  `OpenCVCameraConfig` declares nine fields; the builder read six, so no
  caller could set the connect-time warmup or select an OpenCV backend
  (`cv2.CAP_V4L2`) at all -- the value was accepted and the default compiled.
- Any other key was **discarded**. `heigth=1080` (or `fourc="MJPG"`, or the
  `serial` key the documentation itself advertised) returned a working config
  streaming at the 640x480 default, with no signal that the option had been
  dropped.

The accepted vocabulary is now derived from
`dataclasses.fields(OpenCVCameraConfig)`, so every declared field -- including
fields future lerobot releases add -- is reachable, and an unknown key raises
`ValueError` naming the camera, the unknown option and the closest declared
field, per AGENTS.md > Review Learnings (#86). Three dead-end errors on the same
read path are also resolved: an omitted `index_or_path` reported `KeyError:
'index_or_path'`, a non-mapping camera entry reported `AttributeError: 'str'
object has no attribute 'get'`, and a value lerobot's own validation refuses
(e.g. a 3-character `fourcc`) raised without naming which camera it came from.

The documented `fps=30`/`width=640`/`height=480` defaults are unchanged.
