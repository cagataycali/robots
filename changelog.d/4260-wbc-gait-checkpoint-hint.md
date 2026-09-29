### Fixed: `wbc_gait` names the gait-clock weights it needs when no checkpoint is found

`create_policy("wbc_gait")` without a usable checkpoint inherited the `wbc`
error, which told the user to fetch the non-gait
`GR00T-WholeBodyControl-Balance.onnx` / `-Walk.onnx` pair. Those weights are
516 wide and the gait variant refuses them by shape on load. The error now
names the single 570-wide `policy.onnx` this variant loads and points the
Balance/Walk pair at the `wbc` provider.
