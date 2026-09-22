---
description: Operator approval for mesh actions - which actions are gated, rate limits, payload validation, and how a script answers a command gate.
---

# Command approval

## Operator approval for fleet-wide actions

The `broadcast` and `emergency_stop` actions on the `robot_mesh` tool affect every peer on the network. To keep an agent from issuing fleet-wide commands autonomously (or under prompt injection), both are gated behind a human-in-the-loop interrupt: the Strands runtime pauses the agent loop and asks the operator out-of-band of the LLM's tool arguments. Per-action rate limits, command validation and an audit trail run alongside the interrupt. Outside an agent loop (a bare script or unit test), both actions fail closed.

- **The default gate is broader than just fleet-wide actions.** Out of the box every physical-actuation action is gated: `emergency_stop`, `broadcast`, `tell`, `send`, `stop` and `rpc` (the Device Connect device-native call). A prompt-injected agent therefore cannot drive any physical command - single-peer or fleet-wide - without an explicit operator approval.
- **Approval is an explicit affirmative.** Only `y` / `yes` / `approve` / `approved` count as approval; anything else, including an empty response, is a decline.
- **`STRANDS_MESH_HITL_ACTIONS` tunes the gate.** Widen it to `all` (also gates the read-only `subscribe` / `watch` actions), narrow it to a comma-separated subset, or set it to `none`. `none` re-opens the entire physical-actuation surface to the LLM without confirmation - the SDK logs a one-time warning while it is in effect. Do not use it outside a fully trusted, non-networked test.
- **The prompt states what it verified about the target, and the verdict is not an authorisation.** The gate is per-action, so it asks about a peer without knowing what that peer is. It reports `peer_is_physical`'s reading of the peer's presence - `it reports real hardware (so101_follower)`, `it reports itself as sim`, or `it is not on the fleet snapshot, so it cannot be shown to be a sim` - and fails closed: a peer is metal unless its presence shows it is a sim. The same verdict is carried as `physical` / `verified` in the interrupt's structured reason, so a host UI cannot disagree with the operator's sentence. It does not change which actions are gated: a `tell` aimed at a sim still stops and asks. `robot_type` and `world` arrive over the wire from the peer itself and presence authenticates neither, so a self-report is fit to tell an operator what a peer says about itself and unfit to stand in for the operator.
- **Rate limits bound LLM-driven nuisance** independently of approval: `emergency_stop` at 3/min, `broadcast` at 10/min, `tell`/`send` at 30/min. A declined approval does not consume a slot, so declining nuisance prompts can never lock an operator out of a genuine emergency stop. A slot is reserved atomically at the point the action is known to run, so concurrent invocations cannot exceed the cap - which matters most when `STRANDS_MESH_HITL_ACTIONS` narrows the gate and the cap is the only bound left.
- **Audit trail.** Every `tell` / `send` / `broadcast` / `stop` / `emergency_stop` / `rpc` - and every approval, decline, validation rejection and rate-limit rejection - is written to the [audit log](audit-log.md). The read-only actions are recorded too (`peers`, `status`, `subscribe`, `watch`, `inbox`, `unsubscribe`), on whichever backend served them, so the log records what the agent *read* about the fleet as well as what it told the fleet to do - `peers` returns every device id and function name the fleet exposes, which is the callable surface a later `rpc` would use.

Reference: `strands_robots.tools.robot_mesh`.

## Payload validation

`validate_command` bounds every field of an incoming `execute` / `start` command before it reaches the dispatcher: the action must be in the allowlist, `duration` and `policy_port` are range-checked, `policy_host` / `server_address` / `model_path` are allowlist-gated, and every string field is length-bounded and refuses C0/DEL/C1 control characters. That last check is why the audit trail is worth keeping: `instruction` is free-form text from a remote peer, so admitting a CR or LF would let one log call emit two records and let the second impersonate a different level and logger. Natural-language fields bound only the control range - a non-ASCII instruction is admitted - while identifier fields such as `robot_name` stay printable-ASCII-only.

Reaching that validator at all means decoding the caller's `command` string, and `json.loads` refuses one three ways: malformed syntax, a number wider than `sys.get_int_max_str_digits()`, and a document nested past the interpreter stack (a `RecursionError`, which is not a `ValueError` at all). All three are decided locally and reported through the tool's `{"status": "error"}` envelope with an audit row, so a `broadcast` refused for its body is still a record.

### Locomotion velocity envelope (target_velocity)

`validate_command` holds every component of a mesh `execute` / `start` payload's `target_velocity` to one locomotion envelope: +-2.0 m/s for the linear components (`vx`, `vy`, and any component past the third) and +-2.0 rad/s for `omega`. The WBC policy applies the same bound again before the value is scaled into its observation, from the same definition (`strands_robots.locomotion_envelope`), so a caller that reaches the policy without the mesh meets the same refusal. Out of envelope is refused with a reason naming the component, value, bound and unit - never clamped, because a clamped `[1e6, 0, 0]` is still a sprint the caller did not command. Operators with a faster platform raise the bounds with `STRANDS_MAX_TARGET_LINEAR_VELOCITY_MPS` / `STRANDS_MAX_TARGET_ANGULAR_VELOCITY_RPS` (positive finite floats; anything else leaves the default in force; re-read on every call, no restart).

### Policy vocabulary allowlist (policy_type / policy_provider)

`validate_command` gates every mesh `execute` / `start` payload's `policy_type` and `policy_provider` fields against a built-in allowlist. The two vocabularies share one allowlist by design: `policy_type` names a LeRobot policy *family* (`act`, `diffusion`, `pi0`, `smolvla`, ...) and `policy_provider` names a spelling this package's `create_policy` resolves (`groot`, `wbc`, `moveit`, `microduck`, ...). A provider or family outside the list is refused on the mesh path with the offending value named, rather than surfacing as a silent availability bug.

- `STRANDS_MESH_POLICY_TYPE_ALLOW` - optional; comma-separated extras appended to the built-in list. It widens both `policy_type` *and* `policy_provider` at once, because a payload naming a new provider generally also names a new family. Each entry is charset-validated against `^[a-z][a-z0-9_]*$` at parse time; a malformed entry drops with a WARNING naming both this variable and the offending token, rather than widening the allowlist silently. Spellings are normalised through `.lower()`, so `"FOO,BAR"` matches a payload naming `foo`.
- Widening this set relaxes no other gate. `policy_host` (host / CIDR allowlist), `server_address`, `pretrained_name_or_path` (HuggingFace repo allowlist under `trust_remote_code`) and `model_path` (path-traversal charset) are still enforced against every widened payload. This variable answers *which providers the mesh knows about*, not *which endpoints they may reach*.

**Do not use this variable to work around a registry omission.** Adding a provider to `registry/policies.json` must include the corresponding edit to `_REGISTRY_POLICY_PROVIDERS` in `mesh/security.py`, and a guard test refuses any registry spelling that set omits, so the omission fails CI rather than shipping as a mesh-only availability bug. This variable is for extending the vocabulary *beyond* what the registry knows.

Reference: `strands_robots.mesh.security._policy_type_allowlist`, `strands_robots.mesh.security._REGISTRY_POLICY_PROVIDERS`, `strands_robots.mesh.security._LEROBOT_POLICY_FAMILIES`.

## Answering a command gate from a script

An interrupt is a paused run, not a result: `agent(...)` returns with `result.stop_reason == "interrupt"` and the question in `result.interrupts`, and nothing has moved. Every interrupt raised through the shared command gate - `Robot(mode="real")`, `serial_tool`, `pose_tool`, `use_unitree`, the ROS transports - carries the same structured `reason`: `action`, `target`, `warning` (what would move, ending in `Reply 'y' to approve`), and `how_to_answer`, which states that the call is paused, the exact resume form, and the exact `..._COMMAND_ALLOW=<value>` that would pre-approve this same command with no operator present. It is in the `reason` because `print(result)` on a paused run prints that dict: the script that hit the gate is holding the remedy.

```python
result = agent("Rotate the wrist 5 degrees.")
while result.stop_reason == "interrupt":
    for question in result.interrupts:
        print(question.reason["warning"])         # what would move
        print(question.reason["how_to_answer"])   # how to answer it, and what pre-approves it
    result = agent(
        [
            {"interruptResponse": {"interruptId": question.id, "response": input("approve? [y/N] ")}}
            for question in result.interrupts
        ]
    )
```

Only `y` / `yes` / `approve` / `approved` approve; anything else declines and nothing moves. The value `how_to_answer` names is the one this tool's own allowlist matcher accepts, which is not the same shape for every tool - an action name for `Robot` / `serial_tool` / `pose_tool`, a `service.operation` pair for `use_unitree`, a surface name for the ROS transports - so it is read from that matcher rather than described.
