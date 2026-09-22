---
description: Security considerations for deploying Strands Robots past a trusted lab - prompt injection, refusal codes, secrets and telemetry exposure, with one page per surface.
---

# Security considerations

Strands Robots actuates machines in physical space, pulls models and datasets from the network, runs containers, and coordinates fleets. Work through the surface below before any configuration leaves a trusted lab.

| Page | Surface |
| --- | --- |
| [Mesh authentication](security/mesh.md) | Zenoh auth modes, mTLS material, operator ACLs, fleet namespace, AWS IoT, audit log |
| [Audit log](security/audit-log.md) | Where fleet actions are recorded, rotation, per-record HMAC |
| [Command approval](security/commands.md) | Operator interrupts on mesh actions, payload validation, answering a gate from a script |
| [Hardware and host access](security/hardware.md) | Real-mode robots, serial and pose tools, the ROS 2 command surface, the asset cache |
| [Remote policy code](security/policy-code.md) | `trust_remote_code` checkpoints, GR00T inference containers |

!!! danger "Do not report vulnerabilities here"
    Do not open a public GitHub issue for security concerns. Report via the AWS Vulnerability Disclosure Program on [HackerOne](https://hackerone.com/aws_vdp) or email [aws-security@amazon.com](mailto:aws-security@amazon.com). See [SECURITY.md](https://github.com/strands-labs/robots/blob/main/SECURITY.md).

## Prompt injection

Untrusted data fed to an agent can be treated as instructions. These agents actuate machines, so treat every one of these as untrusted input: the operator prompt, task instructions broadcast over the mesh, camera and observation text surfaced back into context, dataset metadata, and checkpoint descriptions pulled from the Hub.

The controls the SDK already provides, and which you should rely on rather than disable:

- **Tool scoping.** The strongest mitigation is giving an agent only the tools its task needs. An agent that never receives `robot_mesh`, `serial_tool` or `Robot(mode="real")` cannot be coerced into a fleet broadcast, a raw serial write or a physical actuation whatever the injected text says.
- **Out-of-band human approval for physical actuation** ([command approval](security/commands.md)). The approval is delivered outside the LLM's tool-argument flow, so an injected prompt that sets an "approved" flag in the command body cannot bypass the gate.
- **Operator approval for ROS 2 command surfaces.** `use_ros`, `use_rtps` and `use_rosbridge` each gate the verbs that can carry a command - a topic `publish`, a `service_call`, an `action_send_goal` - against a blocklist of safety-critical surfaces. Reading the same surfaces stays ungated. See [the ROS 2 command surface](security/hardware.md#ros-2-dds-bridge-command-surface) and [safety-critical command surfaces](ros2/safety.md#safety-critical-command-surfaces-need-operator-approval).
- **Payload validation** of every mesh command, so an injected instruction cannot smuggle an out-of-bounds duration, an attacker-controlled inference host or an arbitrary model path. See [payload validation](security/commands.md#payload-validation).

## Refusal codes are the stable contract; prose is not

Some refusals are *continuable*: the request was well formed, and an operator who accepts the risk can grant something that makes the identical request succeed - an untrusted policy provider, a repo or host or policy type outside a mesh allowlist, a teleop frame past the value envelope. Anything that offers that answer (a consent card, an approval endpoint, a supervising agent) has to recognise which refusal it is looking at.

Recognise it by its `code`, never by its message text:

```python
from strands_robots import refusal_codes
from strands_robots.mesh.security import ValidationError, validate_command

try:
    validate_command(cmd)
except ValidationError as refusal:
    if refusal.code == refusal_codes.HF_REPO_NOT_ALLOWED:
        offer_to_allowlist(refusal.subject)          # the repo, already parsed out
        print(refusal_codes.REFUSAL_GRANTS[refusal.code])   # STRANDS_MESH_HF_REPO_ALLOW
```

- **`code` is stable; the message is not.** `code` is a member of `refusal_codes.REFUSAL_CODES`, a closed vocabulary you may switch on. The message is an operator-facing sentence and may be reworded at any time, so matching on prose couples you to wording that is free to change.
- **`subject` is what the refusal is about**, so you do not parse it back out: the repo id, the host or whole `server_address`, the policy type or provider name, the joint key, the refused provider.
- **`REFUSAL_GRANTS` names the environment variable that lifts each refusal**, so your consumer and this package cannot drift apart. The variable is half the answer - what to set it to differs by code. `HF_REPO_NOT_ALLOWED`, `POLICY_TYPE_NOT_ALLOWED` and `POLICY_HOST_NOT_ALLOWED` are allowlists you add `subject` to; `TRUST_REMOTE_CODE_REQUIRED` is a flag you set to `1`; `TELEOP_VALUE_OUT_OF_RANGE` is a bound you raise above the refused magnitude. Each code states its own operation in `refusal_codes`.
- **A refusal with no code is not continuable.** `code` is `None` for rejections an operator cannot grant their way past - a schema failure, an over-long instruction, a lockout. Treat `code is None` as "show the message and stop", not as an unknown code.
- **Codes are additive.** Switch on the codes you know and fall through to the message for the rest.
- **Every code you receive is in `REFUSAL_CODES`.** Nothing validates `code` at runtime, so what backs the closed vocabulary is a static scan over every raise site in the package. `REFUSAL_GRANTS[refusal.code]` is therefore safe for any code you are handed: a code outside the vocabulary is a defect in this package.

The in-tree consumer is `strands_robots.dashboard.consent.classify_refusal`, which builds the dashboard's consent card from `code` and `subject` alone.

Reference: `strands_robots.refusal_codes`; `strands_robots.mesh.security.SecurityError`; `strands_robots.policies.factory.UntrustedRemoteCodeError`.

## Credentials and secrets

Handle each class per least privilege:

- `HF_TOKEN` is only needed to push datasets or pull gated checkpoints, and should be scoped to write only when you push. The default sim and Mock path needs no token - do not export one where it is not required.
- AWS credentials drive the Bedrock model provider; scope them to the Bedrock model and region in use.
- mTLS certificates and AWS IoT provisioning material are fleet-wide secrets - provision per device, store securely, rotate and revoke on decommission.

Avoid baking any of these into images, example scripts or notebooks.

## Telemetry exposure to the agent context

The `subscribe` / `watch` actions pull mesh telemetry into the LLM context. By default they are restricted to low-impact, fleet-shared topics (presence, health, safety); subscribing to another peer's command, state, camera or input streams is blocked, with the transport ACL as the primary control and the tool-layer allowlist as defence in depth. If you extend `STRANDS_MESH_SUBSCRIBE_ALLOW`, avoid wildcards that would let the agent observe - and exfiltrate into its context - another peer's control or sensor streams.
