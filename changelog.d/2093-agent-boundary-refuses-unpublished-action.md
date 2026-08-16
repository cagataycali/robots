### Changed: an action the MuJoCo tool schema does not advertise is refused at the agent boundary

`_dispatch_action` resolves an action with `getattr` and no allowlist, so every public
method of the engine is dispatchable while `tool_spec.json`'s `action` enum advertises a
curated 77-entry subset. That width is a deliberate Python convenience, but it reached a
model as well: the two agent-facing entry points - `__call__`, the form the README markets
as `robot(action="...")`, and `stream`, which the agent runtime drives - forwarded any name
straight to the router. So the 23 dispatchable-but-unadvertised capabilities were invocable
by a model that was never told they existed, including the `TeleopMixin` cluster, which
drives real hardware from a host input device.

Both entry points now refuse a non-enum action before dispatching, and the refusal
distinguishes the two cases a caller needs told apart: a name that resolves to no method
stays `Unknown action`, while one that resolves to a method held back from the enum reports
that it exists and is reachable from Python only. Previously the second case ran, and its
own error message could recommend a further unadvertised action - `stream(action="teleoperate")`
answered `No teleoperators attached. Use attach_teleop() first.`, coaching the model toward
a second capability its schema never advertised.

`_dispatch_action` keeps its full width: the refusal lives at the boundary, not in the
router, so direct Python callers are unaffected and every held-back capability stays
reachable through its own method. The allowlist is derived from the schema rather than
restated beside it, so the published set cannot drift from what the model is told exists.

Resolves the remaining question in #2093; the inventory side landed earlier in #2104.
