# Security

At the end of this page you can name every control between a language model and a moving robot, the environment variable that widens or narrows each one, and where the evidence is written.

The posture in one sentence: an agent may read anything; a command that can move a robot stops for an operator, travels only on an authenticated wire, is validated against allowlists on both ends, and leaves a signed row behind.

## The layers

| layer | what it decides | where |
|---|---|---|
| operator gate | whether this command may move a robot at all | `_command_gate.gate_motion`, [agents](agents.md) |
| motion grants | that a browser "yes" is spent once, for the exact tool input it approved | `_motion_grants` |
| command validation | that a mesh command names an allowed action, provider, host and checkpoint | `mesh.security.validate_command` |
| wire auth | who may publish on the mesh | mTLS, `STRANDS_MESH_AUTH_MODE` |
| ACL | which certificate may publish or subscribe on which key | `STRANDS_MESH_ACL_FILE`, [bridges](mesh/bridges.md) |
| transport caps | how many bytes and how often, before deserialisation | `STRANDS_MESH_MAX_*`, `STRANDS_MESH_*_RATE_HZ` |
| safety envelopes | that an e-stop or resume is fresh, unreplayed and, for resume, proven | [safety and e-stop](mesh/safety-and-estop.md) |
| driver gates | that the hardware is in a state where a write is safe | each driver, [drivers](hardware/drivers.md) |
| bus access | that one caller at a time owns a serial bus | `bus_access` |
| path validation | that a tool writes only where it was told | `_path_validation` |
| audit log | what happened, in order, tamper-evident | `audit`, `_hitl_audit` |

## The operator gate

Every path from a model to an actuator ends in `gate_motion`: allowlist variable, then `BYPASS_TOOL_CONSENT=true` (a WARNING), then a Strands interrupt the operator answers out of band, failing closed when nobody can be asked. Reading is never gated. Stopping is never gated. The operator's reply never reaches the model; it goes to the audit log. The verb tables and the exact refusal text are on [agents](agents.md).

The gate lives at the package root because six tools and the hardware `Robot` call it; the ROS blocklist is one set for three transports because it describes a physical surface; a refusal names the value that would pre-approve the call (`STRANDS_POSE_COMMAND_ALLOW=move_motor`), never `=true`.

## Allowlists on the mesh

`validate_command` runs on the sender and the receiver. Beyond `ALLOWED_ACTIONS`, an `execute` may only name a `policy_provider` or `policy_type` in the shipped set plus `STRANDS_MESH_POLICY_TYPE_ALLOW` (the two share one allowlist; a provider missing from `registry/policies.json` belongs in the registry, not the variable), a policy server host in `localhost`, `127.0.0.1`, `::1` plus `STRANDS_MESH_POLICY_HOST_ALLOW`, and a checkpoint under a Hub org in `nvidia`, `huggingface`, `lerobot` plus `STRANDS_MESH_HF_REPO_ALLOW`. Local checkpoint paths are refused on the wire. Each variable's entries must be lowercase identifiers (`^[a-z][a-z0-9_]*$`), so `STRANDS_MESH_HF_REPO_ALLOW="nvidia,;rm -rf /"` is a refusal, not an allowlist. Silent defaults are not honoured on the boundary: an `execute` without a provider is refused before it leaves the process.

Teleoperation frames are bounded too: `STRANDS_MESH_INPUT_VALUE_ABS` (720), `STRANDS_MESH_INPUT_SLEW_ABS`, `STRANDS_MESH_INPUT_MAX_HZ` (100), and the local loop's `STRANDS_TELEOP_SLEW_ABS` (500). An over-speed frame is refused and counted, never clamped toward the command.

## The wire

`STRANDS_MESH_AUTH_MODE=mtls` is the default and cannot be turned off without a second factor (`STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1`, or the one-machine preset `STRANDS_MESH_LOCAL_DEV=true`). Multicast discovery is off by default. A permissive ACL under mTLS refuses to start until acknowledged. The IoT leg binds each robot's X.509 CN to its Thing name and scopes its topics with `${iot:Connection.Thing.ThingName}`; a direct reply goes only to `strands/<sender>/response/<self>/<turn>` ([direct](mesh/direct.md)). The pure-RTPS ROS 2 bridge refuses an inbound command surface without DDS Security unless `STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1` ([ROS 2](ros2.md)). rosbridge is unauthenticated by design, for trusted networks.

## Paths, buses, subprocesses

`validate_save_path` refuses a write into `/etc/`, `/usr/`, `/dev/`, `/proc/` and their macOS and Windows equivalents, and `resolve_output_path` refuses a file name that leaves the directory it was given; every tool that writes a caller-supplied path (`lerobot_camera`, `reachy_camera`, the judge's `write_label`, training) runs both. Dataset ids, bucket names and `run_id`s that reach the `hf` CLI are matched against allowlists before any subprocess. `use_lerobot` refuses `lerobot.scripts`, `push_to_hub`, `upload_folder` and `save_to_disk` by name so a prompt-injected call cannot push or spawn training. `lerobot_train`'s `extra_flags` are gated by `STRANDS_TRAIN_EXTRA_FLAGS_ALLOW`; GR00T container images by `STRANDS_GR00T_IMAGE_ALLOW`.

`bus_access.bus_lock` is an `RLock` on the device, held by every reader (state probe, camera publisher, sensors, IoT offload), the teleop writer and a rollout, so a serial bus is one conversation.

## The dashboard

Loopback by default; a non-loopback bind is refused without a passkey or a static token. Every route but the login screen and `/api/health` takes `require_session`. Details on [dashboard](dashboard.md).

## Evidence

`~/.strands_robots/mesh_audit.jsonl` (`STRANDS_MESH_AUDIT_DIR`, mode `0600`) is the one trail: every operator verdict (`llm_tool_action`), every e-stop and resume event, every command refused under lockout, every mesh tool refusal. Rows carry a monotonic `seq` and, with `STRANDS_MESH_AUDIT_PSK`, an HMAC-SHA256 `sig`; a degraded write says so in `sig` (`PSK_DEGRADED`, `SIGN_FAILED`, ...) instead of leaving a hole, and `verify_audit_integrity()` reports edited rows, gaps and unsigned rows. Rotation by `STRANDS_MESH_AUDIT_MAX_BYTES` and `STRANDS_MESH_AUDIT_MAX_FILES`.

When the file cannot be written, `log_safety_event` logs `[audit] failed to write` at WARNING and swallows the error: the peer keeps running with the trail off, so monitor that line.

## Reporting

Refusal codes are the stable contract; prose is not: [reference/refusal-codes](../reference/refusal-codes.md). Vulnerabilities go to `SECURITY.md`, linked from [project/security-policy](../project/security-policy.md).
