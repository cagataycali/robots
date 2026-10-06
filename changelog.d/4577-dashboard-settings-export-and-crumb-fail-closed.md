### Fixed: a dashboard settings value no longer reaches a gate-bearing variable, and the record crumb stays in the service directory

`settings.apply_mesh_env()` copied `mesh.policy_type_allow` and
`runtime.trust_remote_code` into `STRANDS_MESH_POLICY_TYPE_ALLOW` and
`STRANDS_TRUST_REMOTE_CODE`, the two gates the page's env write already
refused. It now exports only the transport knobs (`ZENOH_CONNECT`,
`ZENOH_LISTEN`, `STRANDS_MESH_PORT`, `STRANDS_MESH_BACKEND`,
`STRANDS_MESH_CAMERA_HZ`) and logs the rest; grant the other two on the host or
through the consent panel. `STRANDS_DASH_RECORD_CRUMB` must point inside
`~/.strands_dashboard` (otherwise the default path is used), and it and
`STRANDS_ROBOTS_VIDEO_ROOT` are shown but no longer page-writable. The page's
env write list is now its own explicit set rather than the display list.
