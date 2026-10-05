### Fixed: a Microduck `unknown skill` refusal names the near-match or the policy surface

`send_action({"skill": ...})` and the `do` verb now end an `unknown skill`
refusal with `Did you mean: 'ball_kick_left' -> 'kick_left'?` when the name is
close to one the robot lists, and otherwise point at `create_policy("microduck",
onnx_path=...)`, where the ONNX actors on the policies page (`alpha_walking`,
`alpha_stand`, ...) actually run.
