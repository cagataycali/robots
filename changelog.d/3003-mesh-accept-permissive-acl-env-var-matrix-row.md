### Docs: `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` is named on the README matrix and the security page

`STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` is the acknowledgement token that lets a
blacklist-shaped operator ACL load.  `strands_robots.mesh._acl_config._validate_acl_shape`
raises `PermissiveACLError` on ACL load when `STRANDS_MESH_ACL_FILE` points at
a file whose `default_permission` is `"allow"` and whose `rules` list is
non-empty, unless this variable is set to `1`/`true`/`yes` -- so on a fleet
that supplies an operator ACL of that shape, this variable is the difference
between a mesh that opens a session and one that refuses at load time.

Before this change: the README env-var matrix named 32 other `STRANDS_MESH_*`
variables and 0 rows for this one, and `docs/security.md` named the variable
in one bullet but framed it as a warning-silencer for the built-in permissive
default -- a *different* posture from the one the module implements.  So an
operator tracing the refusal from `_acl_config.py` found the variable, and an
operator tracing it from either documentation surface did not, or was pushed
toward setting it to silence a warning that never fires for the case they
were configuring.

Fix: one README matrix row beside `STRANDS_MESH_ACL_FILE`; one
`### Blacklist ACL acknowledgement` subsection in `docs/security.md` between
the namespace and policy-vocabulary sections, correcting the framing and
distinguishing the two ACL shapes (`allow` + rules is blacklist, `deny` +
rules is whitelist); one 11-cell derived guard mirroring the shape of the
sibling `test_docs_mesh_*_env_var_reference.py` guards (audit / TLS /
namespace / policy-type-allow).  The guard derives its population from the
module's own `os.getenv` literals so a future sibling variable is held to the
same documentation rule the hour it lands.

Continues the four-shipment sequence from fires 131/138 (`STRANDS_MESH_TLS_*`
#2945, `STRANDS_MESH_AUDIT_*` #2979, `STRANDS_MESH_NAMESPACE` #2985,
`STRANDS_MESH_POLICY_TYPE_ALLOW` #2990).  Refs harness#376.
