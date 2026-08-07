### Quality: the WBC tests pin that the instruction string is discarded, instead of only saying so

`WBCPolicy.get_actions` and `WBCGaitPolicy.get_actions` each take an
`instruction` and read no goal from it — the locomotion command comes from the
well-known `target_velocity` / `target_orientation` / `height` kwargs (plus
`gait_frequency` on the gait variant). Both docstrings say "`instruction` is
ignored". Neither said it in a test.

On these two providers the discard is not on a parameter a caller has to go out
of their way to reach — on the hardware path it is the **only** thing a caller
supplies. `HardwareRobot._run_policy_loop` calls
`get_actions(observation, instruction)` with no kwargs at all, and the mesh
dispatcher that feeds it *requires* a non-empty one: `validate_command` refuses
an `execute` / `start` payload without it, and `wbc` is on the
`is_safe_policy_provider` allowlist the same validator enforces. So a peer drives
WBC by sending words, the words are mandatory, and every goal component is then
taken from the config instead — `"walk forward"` is accepted, the robot does
whatever `target_velocity` defaults to, and the result reports success.

Nothing pinned that. No `get_actions` call anywhere in the WBC suite passed a
non-empty instruction, so the claim held **vacuously**: wiring the instruction
into the goal resolution would have turned both docstrings into lies with every
test still green.

Refusing a non-empty instruction is not the remedy — every `Policy` accepts one
and the runner forwards it verbatim to the providers that do read it, `curobo`
among them — so the discard stays and a test keeps it honest. Acceptance is
pinned alongside it.

For each provider the pin drives one tick on a fresh policy per probe
instruction and asserts the instruction reaches neither the observation array
handed to the ONNX session nor the returned action dict, and does not move
`WBCPolicy`'s walk-vs-main session choice. Non-vacuity is asserted in-suite
rather than assumed: the same probe values supplied through the documented kwarg
spellings are shown to move every one of those channels, so the equality
assertions cannot pass by comparing a constant. Two probes are written in
spellings something in this tree really parses — a JSON object of the shape
`curobo` accepts, and a `key=value` pair — so they are not inert prose.
