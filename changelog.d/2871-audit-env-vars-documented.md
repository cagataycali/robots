### Docs: the audit log's environment variables are named on the security page

`strands_robots/mesh/audit.py` reads four `STRANDS_MESH_AUDIT_*` environment
variables that decide where the mesh's tamper-evident audit trail is written,
how large it may grow, and whether it carries a per-record HMAC:
`STRANDS_MESH_AUDIT_DIR`, `STRANDS_MESH_AUDIT_PSK`,
`STRANDS_MESH_AUDIT_MAX_BYTES`, and `STRANDS_MESH_AUDIT_MAX_FILES`.  None of
the four appeared anywhere under `docs/`.  An operator who read
`docs/security.md` believed the transport was hardened and the mesh action
surface was gated, and never saw the four variables that decide whether the
log they were relying on to reconstruct fleet-wide actions was itself
forgeable and where on disk it lived.

The failure was silent by construction: the module wrote the log at
`~/.strands_robots/mesh_audit.jsonl` whether or not the PSK was configured,
so `ls` on the default directory showed a well-formed JSONL trail on both
sides of the tamper-evidence contract.  Only `verify_audit_integrity` could
tell the two apart, and only if invoked - and the environment variable that
turned the check on was undocumented, so a reader with a hardening checklist
had nothing to check.

`docs/security.md` gains an "Audit log" section, sibling to the transport
posture sections above and to "Credentials and secrets" below, listing all
four variables in the form the page already uses for every other
configuration knob: whether the variable is required, what it defaults to,
and what setting it wrong looks like at runtime.  The section also names the
symlink refusal, because a hardening reader who set `_DIR` to a controlled
path expects to know that a symlink already sitting at the target is
refused before the daemon starts writing.

The rule is pinned by
`tests/mesh/test_docs_audit_env_vars_reference.py`, following the same
AST-derived pattern as `tests/test_docs_device_connect_env_reference.py`
and #2870's `tests/mesh/test_docs_iot_credentials_reference.py`: every
`STRANDS_MESH_AUDIT_*` name the module reads must appear on the page, all
must appear under one heading, and that heading must be the "Audit log"
heading the module docstring references.  A fifth variable added to
`mesh/audit.py` later is graded on arrival.  This is the first
scoped-and-shipped answer to the guardrail called for by harness#376
("~58 `STRANDS_*` vars undocumented"); the same test pattern extends to
the other groups (TLS, transport, bridge, training, sim) named in that
issue.
