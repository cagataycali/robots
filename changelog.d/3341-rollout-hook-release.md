### Fixed: a recorded Isaac or Newton rollout leaves the robot idle when it ends

Three simulation backends raise the per-robot `policy_running` flag inside
`_make_run_policy_hook`, the seam that layers dataset recording onto the shared
run-policy loop. That flag is what a backend's motion primitives and busy
guards refuse on, so it has to come back down when the rollout ends -- the
guarantee `MuJoCoSimEngine._drive_rollout` states, and reaches through its own
`run_policy` override.

Isaac and Newton have no such override. Their hook builders are independent
copies of the MuJoCo one: both kept the raise and neither had anywhere to put
the release, so a recorded rollout marked the robot as driven for the rest of
the session. On Isaac every motion primitive then answered `Cannot 'set_gripper'
on 'so100' while its policy is running ... Wait for the rollout to finish (Isaac
policy loops clear the flag on exit)` and `run_multi_policy` answered `policy
already running on 'so100'. Stop it first` -- two remedies for a rollout that
had already ended, neither of them reachable, since Isaac exposes no
`stop_policy`. On Newton the stale flag became `SimRobot.request_policy_stop`'s
`was_running=True`, which is the entire verdict of the stop paths that reach a
backend without `stop_policy`, so the Device Connect `stop` RPC reported a
halted rollout on an idle simulation.

The release now has one owner. `SimEngine._release_run_policy_hook` is the other
half of the hook seam, called in a `finally` around every rollout the facade
drives -- the single-episode `run_policy` path and each episode of a
multi-episode run -- so a rollout that ends for any reason (completion, a
cooperative stop, or a raise) leaves the robot idle. Isaac and Newton implement
it beside their hook builders; MuJoCo, which already owned the release, is
unchanged.
