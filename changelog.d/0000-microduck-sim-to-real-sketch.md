### Fixed: the Microduck hardware page's sim-to-real sketch runs the ONNX walker it names

`docs/learn/hardware/microduck.md` passed the robot name positionally to
`run_policy`, which binds `robot_name` and leaves `policy_provider` at its
`"mock"` default, so the sketch that follows "the on-robot policy is the same
`alpha_walking.onnx`" ran `MockPolicy`. It now names `policy_provider="microduck"`,
the weight, and the `target_velocity` twist the policy reads (it does not read
the instruction). A docs grader refuses any documented `run_policy` call that
leaves its policy to the default.
