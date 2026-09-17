### The approval prompt says how long the arm may move, and whether the words matter

The one sentence an operator answers before a real arm moves read `'execute'
drives the real robot 'so101' with 'Wave the arm' (policy mock at
localhost:None); it needs operator approval before it is dispatched`. Two facts
that decide the answer were missing: with `mock` the arm does not wave - every
joint follows a sinusoid whatever the task says, so the operator approved a
motion they were not told about - and the wall-clock budget the control loop
honours went unstated, so "y" bought an unknown length of that motion.

It now reads `'execute' drives the real robot 'so101' for up to 3s with 'Wave
the arm' (policy mock built in this process, no server); it needs operator
approval before it is dispatched. Note: MockPolicy does not read the
instruction. Its actions - a test motion on every joint - are commanded to the
robot whatever the task says; no status or completion that follows will mean
the task was performed.` (The policy phrase is #3761's; this change adds the
budget and the notice around it.)

The budget is the horizon `duration` sets, defaulted like the entry points
(30s), and dropped rather than crashing for a value `_pre_gate_error` has
already refused. The notice is the one `start` and a RUNNING `status` carry, from
the same helper and keyed on the class the provider resolves to, so an operator
reads before dispatch exactly what the envelopes report after - and a policy
that does act on its instruction adds nothing. The headless refusal carries the
same sentence.
