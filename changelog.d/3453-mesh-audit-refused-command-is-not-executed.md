### Fixed

- `Mesh._exec_cmd` audits a handler refusal (`{"error": ...}` / `status == "error"`, e.g. a resume with a bad override code) as `command_refused` with the handler's error instead of `command_executed`, so the safety trail no longer shows a denied resume as an executed one.
