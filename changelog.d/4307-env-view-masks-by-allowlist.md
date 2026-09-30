### Fixed: the dashboard env view hides every value it has not been told is safe to show

`GET /api/config` used to mask an env value only when the key's name contained
a word like KEY, TOKEN or PASSWORD, and returned every other value verbatim, so
`STRANDS_MESH_AUDIT_PSK` (the HMAC key behind the tamper evident audit log) was
disclosed in clear text. The read path now mirrors the write path: a value is
shown in full only for a key on `config_api.SHOWN_ENV_KEYS`, everything else
reports whether it is set, and a mask carries no character of the value (the
old one kept the first three and last two). (f021)
