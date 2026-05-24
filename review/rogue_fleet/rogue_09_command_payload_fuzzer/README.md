# `rogue_09_command_payload_fuzzer`

**AV-09 / AV-10 / AV-11 / AV-12 / AV-13 / AV-14 / AV-15 — payload validator.**

## Threat narrative

Wire-layer defences (mTLS, ACL, low_pass_filter, namespace) all hold,
but an authenticated peer is still in scope: a compromised mesh peer,
a bug in a legit client, an operator running a non-strands tool that
publishes raw payloads. The `validate_command` payload validator is
the last gate before the dispatch table runs an action.

Seven attacks an authenticated peer would try:

1. **Length DoS** — 200 KB instruction string (hosts a 200 KB Python
   string inside the agent's prompt).
2. **Inference pivot** — redirect VLA inference to an
   attacker-controlled host (steals robot frames + actions).
3. **Path traversal in model name** — `../../etc/passwd` as the
   HuggingFace cache subdirectory.
4. **Unallowed HF org** — `evil/backdoor` instead of `lerobot/...`.
5. **24-hour execute** — `duration=86400` to keep the robot stuck
   running an attacker action long after operator intervention.
6. **Non-dict command** — raw string payload (R24-B: pre-fix
   `_dispatch` would crash).
7. **Unknown action** — verb the dispatch table does not know.
8. **Half-spec'd policy** — `policy_type` without
   `policy_provider` (would let an attacker name a registry entry
   with no provider gate).

## Defences in scope

All inside `strands_robots/mesh/security.py`:

* `MAX_INSTRUCTION_LEN`, `MAX_DURATION_S`, `MAX_TIMEOUT_S`,
  `MAX_MODEL_PATH_LEN`
* `_MODEL_PATH_RE` — traversal-safe regex
* `is_safe_policy_host` — loopback default + CIDR/host allowlist
* `is_safe_model_path` — HF-org allowlist (`STRANDS_MESH_HF_REPO_ALLOW`)
* `_DEFAULT_POLICY_TYPES` + `STRANDS_MESH_POLICY_TYPE_ALLOW`
* `ALLOWED_ACTIONS` — the action allowlist
* `validate_command` — enforces all of the above + non-dict reject

## Pass criterion

All seven cases raise (`ValidationError` preferred, any exception
accepted). A silent return for any case is a bypass.

## Cross-references

* Unit-style: `pentest_mesh.py::av_09..av_15`.
* Source: `strands_robots/mesh/security.py`.
