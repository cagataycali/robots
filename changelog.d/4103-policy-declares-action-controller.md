### Changed: a policy declares the action controller it needs

`Policy.requires_action_controller` (class attribute, default `None`) states why a
rollout needs a controller the engine installs. `SimEngine` refuses any policy
declaring one on an engine that cannot install it, instead of knowing
`WBCPolicy` by name; `WBCPolicy` declares the torque shim, which the MuJoCo
engine installs as before. The private hook `_maybe_install_wbc_torque_control`
is renamed `_maybe_install_action_controller`.
