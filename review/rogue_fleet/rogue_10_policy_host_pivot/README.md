# `rogue_10_policy_host_pivot`

**AV-10 / AV-33 — inference target host & address parser hardness.**

## Threat narrative

The VLA policy server is the single most-trusted target a robot
talks to during execution: every camera frame, every action, every
language query passes through it. An attacker who could redirect
that host (via a malformed env var, a parser quirk on `[::1]:8000`,
or a permissive default) would silently MITM the entire control
stack.

Three adjacent surfaces:

1. The host **allowlist** (`is_safe_policy_host`) defaults to
   loopback, extends via `STRANDS_MESH_POLICY_HOST_ALLOW`.
2. The allowlist must reject **malformed env entries** (e.g.
   `;rm -rf /`) without silently fail-opening.
3. The composite **`server_address`** parser handles IPv4, IPv6
   bracketed (`[::1]:8000`), DNS forms, and rejects partial /
   bracket-mismatched inputs.

## What this rogue does

Directly invokes both functions across 12 input shapes; asserts
each hits the expected verdict (allow / block).

## Defences in scope

* `security.is_safe_policy_host` and the `_POLICY_HOST_ENTRY_RE`
  charset gate (F3 fix — malformed entries dropped with WARNING).
* `security.is_safe_server_address` and its IPv6 bracket parser
  (covered in F18-A; `[` without matching `]` rejected).

## Pass criterion

All 12 sub-cases produce the expected boolean.

## Cross-references

* Unit-style: `pentest_mesh.py::av_10_validate_hostile_policy_host`,
  `av_33_ipv6_server_address`.
* Source: `strands_robots/mesh/security.py` -- `is_safe_policy_host`,
  `is_safe_server_address`.
