### Fixed: the real robot's operator-approval prompt describes the motion, not the request

Before an `execute` / `start` moves a real arm, the operator now reads what
will happen - the time budget, whether the policy builds in process or dials
a server, and, for a policy that never reads the instruction (`mock`), that
every joint will follow a sinusoidal test motion whatever the words say. The
prompt used to read `drives the real robot 'so101' with 'Wave the arm' (policy
mock at localhost:None)`: the instruction as the motion, a server that did not
exist, no budget.
