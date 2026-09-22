### Docs: the security guide is six pages, each inside the 1,500-word budget

`docs/security.md` carried the mesh wire, operator approval, host and hardware
access, remote policy code and the audit channel in one 7,978-word page - the
largest on the site, and the one a reader scrolls past to reach the posture they
came for. Split at its H2s into a hub plus five topic pages, each under the
per-page word budget, dropping the rationale and history prose the code and this
log already carry: 7,978 -> 6,506 words.

`security.md` keeps the threat model, prompt injection, the refusal-code
contract, secrets and telemetry exposure, and indexes the rest.
`security/mesh.md` owns the auth modes, mTLS material, ACL postures, namespace
isolation and the AWS IoT transport; `security/audit-log.md` the audit channel;
`security/commands.md` operator approval and payload validation;
`security/hardware.md` the real-mode, serial, pose and ROS 2 command surfaces
plus the asset cache; `security/policy-code.md` the `trust_remote_code` opt-in
and the GR00T inference containers.

Every environment variable, reference symbol and graded claim the page carried is
kept: the reference tests that pin each section read the page that owns it, and
`security.md` leaves the word-budget roster, which can only shrink.
