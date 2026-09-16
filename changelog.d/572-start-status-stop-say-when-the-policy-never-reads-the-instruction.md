### Fixed: the real robot's `start`, `status` and `stop` say when the policy never reads the instruction

A background task on a policy that ignores its instruction (`mock`) used to
report "Task started: 'trace a small circle'", "RUNNING … Steps: 18" and
"Task stopped … Steps completed: 34" with nothing saying the circle was
never attempted; only `execute` carried the notice. All three envelopes now
do - `start` in the present tense, resolved from the registered policy class
before the policy is built (`provider_policy_class`), `status` in the tense
of its state, `stop` in the past. `Policy.reads_instruction` is a class
attribute so the class can answer before an instance exists.
