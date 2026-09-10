### Fixed: `Robot(cameras=...)` resolves the camera `type` through lerobot's registry, so every backend lerobot ships is attachable

`_build_camera_config` imported `OpenCVCameraConfig` unconditionally and refused
any other `type`, while the robot name beside it has always been resolved
generically through lerobot's draccus `ChoiceRegistry`. A camera dict *is* a
serialized `CameraConfig` - `type` is draccus' own choice discriminator and the
remaining keys are fields of the class it selects - so both halves are now
derived: `CameraConfig.get_choice_class(type)` picks the class, and
`dataclasses.fields()` of that class defines the accepted options.
`intelrealsense`, `zmq`, `reachy2_camera` and any installed `lerobot_camera_*`
plugin are attachable; a field no OpenCV config declares (`use_depth`,
`serial_number_or_name`) is reachable for the first time.

This closed a contradiction inside the package: `tools/lerobot_camera` lists and
opens a RealSense, and the real-mode docs name a `realsense_top` camera, but
attaching one to a robot raised `Unsupported camera type`. The refusal now lists
every registered choice and, when the spelling is close, names it - lerobot
registers Intel RealSense as `intelrealsense`, so the obvious guess `realsense`
was a dead end. An absent vendor SDK is still reported as the absent SDK, by
lerobot, where the device is opened.

The choices are populated lazily (`lerobot.cameras.__init__` deliberately
imports no backend, so the registry starts empty), so a `pkgutil` walk mirrors
the one the robot registry already does. The third-party plugin loader both
walks share moved into one cached helper, since a single call registers every
kind at once.
