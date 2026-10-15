### Added: Policy, every LeRobot policy, local or over Zenoh, and the ONNX skills

`Policy("<owner>/<repo>")` (or a local checkpoint directory) runs any LeRobot checkpoint in-process with its saved
processors, and `Policy("zenoh://<host>:<port>/<deployment>")` drives a robot from a `lerobot-policy-server`
deployment over LeRobot's own Zenoh protocol. Joint names, `robot_type`, feature requirements and the checkpoint's
norm mode (degrees or range) are read from the `Description` and the registry, so no robot is named in the code.
`Policy("holosoma")` and `Policy("microduck?skill=<name>")` load ONNX presets: the holosoma G1 walker and ten
microduck skills, fetched from the Hugging Face Hub and checked by sha256. A policy fault stops the run with
`stopped_by="fault"`. New extras: `[onnx]` and `[lerobot]` (Python 3.12+, zenoh below 1.10); LeRobot itself is
installed from a pinned commit until `remote_inference` is on PyPI.
