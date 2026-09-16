### Fixed: a rollout envelope says when the policy never read the instruction it echoes

`Policy.reads_instruction` (default `True`) is `False` on `MockPolicy`, and both
`run_policy` and the real robot's `execute` / `start` envelope append one note
when it is: the policy did not read the instruction, its test motion was
commanded to the robot whatever the task said, and nothing in the report means
the task was performed. `run_policy`'s json payload gains `instruction_read`.
An agent relaying a mock rollout no longer reports the arm "waved" - or, asked
again, that it "stayed still". The real robot's envelope describes an
in-process provider as built in process rather than as running
`on localhost:None`.
