### Fixed: a mistyped camera name gets a close match on every render and record path

`render`, `render_depth`, `render_all` and `start_cameras_recording` answered an
unknown camera with only `Available: [...]`, while `remove_camera` also offered
`Did you mean: ...?` and pointed at `action='list_cameras'`. All of them now
share one message, so a one-letter typo is fixable from the error alone.
