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
path expects to know that a symlink already sitting at the target is refused
rather than written through.

The rule is pinned by
`tests/mesh/test_docs_audit_env_vars_reference.py`, following the same
AST-derived pattern as `tests/test_docs_device_connect_env_reference.py`
and #2870's `tests/mesh/test_docs_iot_credentials_reference.py`: every
`STRANDS_MESH_AUDIT_*` name the module reads must appear on the page, all
must appear under one heading, and that heading must be the "Audit log"
heading the module docstring references.  A fifth variable added to
`mesh/audit.py` later is graded on arrival.  The same test pattern extends to
the other undocumented `STRANDS_*` groups (TLS, transport, bridge, training,
sim).

Two of that section's first-draft claims were the inverse of the module they
document, and both are corrected here.  It promised that a peer which cannot
open the log "refuses to start the auditor, rather than running with the audit
trail silently off"; `log_safety_event` documents the opposite - "write errors
are logged at WARNING and swallowed because an audit-log failure must never
propagate up into the safety code path" - and its write block ends in
`except OSError: logger.warning("[audit] failed to write ...")`.  There is no
eager auditor start at all: `_ensure_paths` runs per write.  Measured over
three ways to make the destination unusable (an unwritable parent, a symlink
at the log path, a directory at the log path), three safety events each time
produced no exception, zero records written, and only WARNING lines - exactly
the posture the sentence promised could not happen.  An operator following it
would have concluded that a running mesh implies a written trail and added no
monitoring for the real signal.

It also named the persisted signature field `hmac`.  The writer assigns
`record["sig"]`, `verify_audit_integrity` reads `record.get("sig")`, and no
record carries an `hmac` key.  The field name is part of the on-disk JSONL
schema, so it is what an external verifier - a SIEM rule, a forensic script -
greps for; a checker built from the old wording finds nothing on a correctly
signed fleet and either alarms permanently or concludes tamper-evidence is
off.  HMAC-the-mechanism was correct throughout; only the field name was
wrong.

Both survived the guard above because a presence rule is satisfied by a
bullet that mentions the variable, whatever the bullet asserts.  The test
file gains the missing axis: the documented signature field is derived from
the writer's own dataflow (the `record[...]` key the `_sign_record` result is
assigned to), every field name the section mentions must be one the record
schema carries, the section may not assert a refusal to start, and every
`[audit]` log line it tells an operator to monitor for must be a string the
module emits.  Behavioural cells hold the two claims against the running
writer, so a rule that stopped deriving is caught by a written record rather
than by agreeing with itself.
