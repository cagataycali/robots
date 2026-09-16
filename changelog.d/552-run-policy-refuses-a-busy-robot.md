### Fixed: `run_policy` and `eval_policy` are refused while another rollout drives the robot

`start_policy` and every joint write pass the per-robot gate (`Cannot ... while
its policy is running. Stop it first`); `run_policy` and `eval_policy` did
not. A second rollout on a robot another thread was already driving ran
concurrently - two policies writing one `ctrl` slice - reported
`Policy complete`, and its `finally` then lowered the `policy_running` claim
and the driver-thread record the first rollout still held. `eval_policy`
never claimed the robot at all.

MuJoCo's `run_policy` now passes the gate before it announces the rollout; the
driving-thread exemption keeps `start_policy`'s worker admitted. `eval_policy`
reads the same gate through a new default seam on the `SimEngine` ABC (`None`
on backends that keep no per-robot claim), so it is refused on MuJoCo and
unchanged elsewhere. A rollout on another robot is still admitted.
