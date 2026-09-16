### Fixed: `run_policy` and `eval_policy` are refused while another rollout drives the robot

`start_policy` and every joint write pass the per-robot gate (`Cannot ... while
its policy is running. Stop it first`); `run_policy` and `eval_policy` did not.
A second rollout on a robot another thread was already driving was admitted, so
two policies wrote one `ctrl` slice, and the second's `finally` then lowered the
`policy_running` claim and the driver-thread record the first still held. That
claim is the running rollout's stop signal, so the first rollout ended early
reporting `Policy stopped` - a stop nobody asked for. A `start_policy` rollout
budgeted 4.0s ran 200 steps to `Policy complete` alone, and 75 steps to
`Policy stopped` when a 0.5s `run_policy` was launched on the same robot at
t=1.0s. `eval_policy` never claimed the robot at all.

MuJoCo's `run_policy` now passes the gate before it announces the rollout; the
driving-thread exemption keeps `start_policy`'s worker admitted. `eval_policy`
reads the same gate through a new default seam on the `SimEngine` ABC (`None` on
backends that keep no per-robot claim), so it is refused on MuJoCo and unchanged
elsewhere. A rollout on another robot is still admitted.
