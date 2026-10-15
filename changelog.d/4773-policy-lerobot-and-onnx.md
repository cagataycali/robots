### Added: Policy, every LeRobot policy, local or over Zenoh, and the ONNX skills

`Policy("lerobot:<repo>")` runs any LeRobot checkpoint in-process with its saved processors, and
`Policy("zenoh://<deployment>")` drives a robot from a `lerobot-policy-server` deployment over LeRobot's own
Zenoh protocol. Joint names, `robot_type`, feature requirements and the checkpoint's norm mode (degrees or
range) are read from the `Description` and the registry, so no robot is named in the code. `Policy("onnx:...")`
loads ONNX presets: the holosoma G1 walker and ten microduck skills, fetched from the Hugging Face Hub and checked
by sha256. A policy fault stops the run with `stopped_by="fault"`. New extras: `[lerobot]` (Python 3.12+) and
`[onnx]`; zenoh is pinned below 1.10.
