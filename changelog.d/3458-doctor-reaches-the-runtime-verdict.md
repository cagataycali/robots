### Fixed: `strands-robots doctor` no longer passes a configuration the runtime refuses

Three rows were measured green while the same environment failed at first use:
`lerobot` imported with a torchcodec that could not load (the first
`LeRobotDataset` then printed the loader traceback and fell back to pyav),
nothing said what `Robot(...).run()` would do about TLS on device-connect, and
`PASS zenoh available` printed while `Robot(mesh=True)` logged
"Mesh did NOT start" and handed back a session that never opened.

The doctor is now a table of `(label, probe)` rows. The new `Torchcodec` row
imports torchcodec's native ops and names the loader's own reason with the
ffmpeg or torch-ABI remedy that fits. The `Device Connect` row reports the
posture `run()` would take through the runtime's own `resolve_allow_insecure`
and `transport_is_authenticated`. The `Mesh` row runs the same auth-mode and ACL
gate as `Mesh.start`, prints its "Pick one" text on refusal (now one shared
`PERMISSIVE_ACL_REFUSAL` constant, so the two cannot name different env vars),
and reports the hub port `open_session` would really bind, each `ZENOH_CONNECT`
endpoint's reachability, and multicast scouting - without opening a session.

The `Mesh` row also reads `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` through the
gate's own predicate. The three runtime gates (`Mesh.start`, the ACL loader,
the per-session warning) each spelled the accepted values `1`/`true`/`yes`
themselves; they now call one `_acl_config.permissive_acl_acknowledged`, and so
does the doctor - which had borrowed a boolean parser that also accepts `on`,
and printed `PASS ... (acknowledged)` for a value `Mesh.start` refused. A
grader now holds the package to one reader of that variable.
