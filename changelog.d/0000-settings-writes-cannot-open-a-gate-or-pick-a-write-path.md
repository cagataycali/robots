### Fixed: a dashboard settings write cannot open the remote-code or policy-type gate, or pick where the record crumb is written

`POST /api/config` and `POST /api/settings` now refuse `runtime.trust_remote_code`
and `mesh.policy_type_allow`, next to `security.auth_token`. Before, either route
stored them and `settings.apply_mesh_env()` exported them as
`STRANDS_TRUST_REMOTE_CODE` and `STRANDS_MESH_POLICY_TYPE_ALLOW` on the next mesh
start, the exact variables the `env` half of the same request refuses.
`config_api.REFUSED_SETTINGS_KEYS` is now derived from the settings schema (every
key whose env spelling is gate-bearing), and `apply_mesh_env` exports none of
them, whatever `settings.json` holds. Grant those gates from the consent card or
set them on the host; clearing one from the page still works. The Settings
drawer's trust checkbox is gone for the same reason.

The page-writable env keys are now their own closed list instead of the display
list. `STRANDS_DASH_RECORD_CRUMB`, `STRANDS_ROBOTS_VIDEO_ROOT` and
`OPENAI_BASE_URL` are still shown but are set on the host. `record_crash.crumb_path()`
ignores, with a warning, an override that resolves outside `~/.strands_dashboard`.
