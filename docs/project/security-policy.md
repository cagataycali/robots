# Security policy

How to report a vulnerability in strands-robots, and what the package itself does about the risks a robot library carries. After this page you know where a report goes and which safety postures the code enforces by default.

## Report a vulnerability

Do not open a public GitHub issue for a security concern. strands-robots is an AWS project and follows the AWS responsible-disclosure process:

- Submit through the AWS Vulnerability Disclosure Program on [HackerOne](https://hackerone.com/aws_vdp), or
- Email [aws-security@amazon.com](mailto:aws-security@amazon.com).

Details are on the [AWS vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). The repository's `SECURITY.md` is the authoritative text.

## What the package enforces

A robot library's security surface is actuation. The postures below are on by default at this commit; each names the module that owns it.

| posture | default | owner |
|---|---|---|
| `Robot(name)` is simulation unless `mode="real"` is spelled out | sim | `robot.py` |
| A call that would move real hardware pauses the agent with the SDK interrupt until an operator answers; `STRANDS_*_COMMAND_ALLOW` pre-approves one command | gated | `_command_gate.py`, `_motion_grants.py` |
| In the dashboard's agent console, stopping is never gated: `stop`, `emergency_stop`, `status` and `peers` bypass the HITL hook. On the mesh tool, `robot_mesh(action="emergency_stop")` is in the default interrupt set; `STRANDS_MESH_HITL_ACTIONS` narrows or widens it | see row | `dashboard/agent_hitl.py`, `tools/robot_mesh.py` |
| Mesh transport authenticates with mTLS; `STRANDS_MESH_AUTH_MODE=none` needs `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` and logs a warning | mTLS | `mesh/_zenoh_config.py` |
| Policy models, hosts and repos outside the mesh allowlists are refused with a [code](../reference/refusal-codes.md); remote code needs `STRANDS_TRUST_REMOTE_CODE=1` | refuse | `mesh/security.py`, `policies/factory.py` |
| `device_connect` admits nobody until an RPC allowlist is set, and refuses a plaintext transport | closed | `device_connect/_authz.py` |
| The dashboard binds loopback only until a passkey is enrolled or `DASHBOARD_AUTH_TOKEN` is set | loopback | `dashboard/cli.py`, `dashboard/auth.py` |
| A G1 joint target outside the per-joint envelope, or a gain outside `[0, KP_MAX]`, is refused, not clamped | refuse | `drivers/g1.py`, `locomotion_envelope.py` |
| Every operator answer and safety event is appended to a sequenced audit log | on | `audit.py`, `_hitl_audit.py` |

Every environment variable named above is in [configuration](../reference/configuration.md). The threat model behind these choices, with the findings that shaped them, is in [learn/security.md](../learn/security.md).

## Dependencies

Transitive packages with a published advisory are floored in `[tool.uv] constraint-dependencies` in `pyproject.toml`, each entry annotated with its GHSA id, and the lockfile parity check in CI keeps `uv.lock` in step. Every third-party GitHub Actions `uses:` is pinned to a commit SHA. Dependabot and CodeQL run on `main`.
