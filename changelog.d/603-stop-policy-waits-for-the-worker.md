### Fixed: `stop_policy` waits for the worker before saying "Stopped"

`stop_policy` lowered the cooperative-stop flag and answered "Stopped on 'x'"
at once, while the worker exited only at its next control tick - so the
caller's next `run_policy` on that robot was refused "while its policy is
running" by a robot the previous answer had just called stopped, and a second
`stop_policy` reported `was_running=true` again. It now joins the worker's
Future (bounded, 2 s) before answering; the `json` block gains
`worker_exited` (`false` when the worker is still blocked in inference, said in
a second sentence). An empty `robot_name` names the rollouts in flight instead
of only "requires 'robot_name'".
