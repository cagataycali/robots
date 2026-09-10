### Changed: the mesh safety subscribers share one envelope path

`Mesh._on_safety_estop` and `Mesh._on_safety_resume` now decode, session-bind and time-gate their envelopes through the same three helpers, and every safety audit record on those paths goes through one never-raising `_audit` wrapper. Refusal messages and audit event names are unchanged; `remote_estop_engaged` / `remote_resume_applied` payloads report `issuer_t` from the validated timestamp instead of re-reading the raw field. No wire or behaviour change otherwise.
