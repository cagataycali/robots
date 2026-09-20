### Fixed: a WBC rollout is refused on a backend that cannot install the torque shim

`WBCPolicy` emits joint-position targets that the stock G1's uniform `kp=500`
servos override, so `run_policy` installs the PD->torque shim
(`WBCTorqueController`) for the call. Only the MuJoCo engine can: the shim is
written against a compiled `MjModel`. Every other backend inherited a no-op
hook, drove the servos directly and reported success - on the Newton backend no
controller was registered for any step and the pelvis sank 0.793 m -> 0.074 m
in one second under `status="success"`, `action_errors: 0`, the fall being the
only report. `SimEngine._maybe_install_wbc_torque_control` now states that
requirement and `run_policy` refuses the rollout before any action is applied,
naming the MuJoCo backend and the `wbc_install_torque_control=False` opt-out,
which still rolls out unchanged for a torque-actuated scene. Towards #3818.
