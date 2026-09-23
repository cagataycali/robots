### Security: an ACL subject bound to `interfaces: ["*"]` is read as wide open, and a missing `json5` refuses the wire instead of raising out of `Mesh.start`

`_validate_acl_shape` refuses a subject that constrains neither `interfaces` nor
`cert_common_names`, and its own message recommends `interfaces: ["*"]` for a
deliberate any-link binding. That shape is wire-identical to the one just
refused (Zenoh reads `["*"]` as `SubjectProperty::Wildcard`), but
`_is_wildcard_subject` tested truthiness, so the recommended escape read as a
constraint and a `deny` + `**`/allow + `["*"]` file slipped the refuse-to-start
gate the validator exists to feed. A dimension is now unconstrained when it is
absent, empty, or carries a `"*"` member.

`_parse_json5` raises `ImportError` when the declared `json5` dependency is
absent (zenoh present, json5 missing). `is_default_acl_in_use`, `snapshot_acl`
and `Mesh._refuse_under_permissive_default_acl` caught only
`(OSError, ValueError)`, and the gate's call site is `try/finally`, so a partial
install left `Mesh.start` as a traceback rather than the documented refusal.
`ImportError` joins the fail-closed tuple at all three sites; the gate now
returns its decision.

`tests/mesh/test_acl_permissive_shape_edge_cases.py` pins `["*"]` alone, `["*"]`
beside real entries, and the `cert_common_names` dimension;
`tests/mesh/test_acl_config.py::TestMissingJSON5IsADecisionNotATraceback` pins
the refusal end to end.
