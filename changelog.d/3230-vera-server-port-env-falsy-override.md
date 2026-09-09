### Fixed

- `VeraConfig` now applies its `VERA_SERVER_PORT` override because it is
  present, not because it is truthy. `VERA_SERVER_PORT=0` was falsy, so the
  override was discarded and the per-embodiment default applied in its place -
  the same value `VeraConfig(server_port=0)` refuses by name, since the client
  cannot learn which ephemeral port the kernel handed the server. A port now
  gets one verdict whichever spelling named it, matching `vis_port` and
  `render_width`, whose overrides already read for presence.
