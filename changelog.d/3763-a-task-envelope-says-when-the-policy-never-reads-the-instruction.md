### Fixed: a task envelope says when the policy never reads the instruction it echoes

`Policy.reads_instruction` (a class attribute, default `True`) is `False` on
`MockPolicy`, and every envelope that echoes an instruction now appends one
sentence when it is: the policy did not read the task, its test motion was
commanded to the robot anyway, and nothing in the report means the task was
performed. The simulation's `run_policy` carries it (and `instruction_read` in
its json block), as do the real robot's `execute`, `start` (present tense,
resolved from the registered policy class before the policy is built),
`status` (in the tense of its state) and `stop`. An agent relaying a mock
rollout no longer reports that the arm waved - or, asked again, that it
"stayed still". `execute` also stops describing an in-process provider as
running `on localhost:None`.
