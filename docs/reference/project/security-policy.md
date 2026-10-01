---
description: Where a vulnerability report goes and which safety postures the code enforces by default.
---

# Security policy

How to report a vulnerability in strands-robots, and what the package does about the risks a robot library carries: where a report goes and which safety postures the code enforces by default.

## Report a vulnerability

Do not open a public GitHub issue for a security concern; strands-robots follows the AWS responsible-disclosure process:

- Submit through the AWS Vulnerability Disclosure Program on [HackerOne](https://hackerone.com/aws_vdp), or
- Email [aws-security@amazon.com](mailto:aws-security@amazon.com).

Details: the [AWS vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/); the repository's `SECURITY.md` is the authoritative text.

## What the package enforces

A robot library's security surface is actuation. These postures are on by default at this commit, each with the module that owns it.

| posture | default | owner |
|---|---|---|
| `Robot(name)` is simulation unless `mode="real"` is spelled | sim | `robot.py` |
| A policy rollout or tool command that would move real hardware pauses the agent until an operator answers (the native drivers' `move_to` excepted at this commit); `STRANDS_*_COMMAND_ALLOW` pre-approves one command | gated | `_command_gate.py`, `_motion_grants.py` |
| Stopping is never gated in the dashboard console (`stop`, `emergency_stop`, `status`, `peers` bypass the hook); on the mesh tool `emergency_stop` is in the default interrupt set, which `STRANDS_MESH_HITL_ACTIONS` narrows or widens | see row | `dashboard/agent_hitl.py`, `tools/robot_mesh.py` |
| Mesh transport authenticates with mTLS; `STRANDS_MESH_AUTH_MODE=none` needs `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` and logs a warning | mTLS | `mesh/_zenoh_config.py` |
| Policy models, hosts and repos outside the mesh allowlists are refused with a [code](../refusal-codes.md); remote code needs `STRANDS_TRUST_REMOTE_CODE=1` | refuse | `mesh/security.py`, `policies/factory.py` |
| `device_connect` admits nobody until an RPC allowlist is set, and refuses a plaintext transport | closed | `device_connect/_authz.py` |
| The dashboard binds loopback until a passkey is enrolled or `DASHBOARD_AUTH_TOKEN` is set | loopback | `dashboard/cli.py`, `dashboard/auth.py` |
| A G1 joint target outside the per-joint envelope, or a gain outside `[0, KP_MAX]`, is refused, not clamped | refuse | `drivers/g1.py`, `locomotion_envelope.py` |
| Every operator answer and safety event lands in a sequenced audit log | on | `audit.py`, `_hitl_audit.py` |

Every variable named above is in [configuration](../configuration.md); the threat model and the findings behind these choices are in [security](../../learn/security.md).

## Dependencies

Transitive packages with a published advisory are floored in `[tool.uv] constraint-dependencies` in `pyproject.toml`, each annotated with its GHSA id; the lockfile parity check keeps `uv.lock` in step. Every third-party GitHub Actions `uses:` is pinned to a commit SHA. Dependabot and CodeQL run on `main`.
