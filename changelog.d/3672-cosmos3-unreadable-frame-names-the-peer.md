### Fixed

- **`policies/cosmos3`**: a WebSocket frame the vendored msgpack+NumPy codec
  cannot read is now reported as a `ConnectionError` naming the endpoint, which
  read it answered, what the codec could not do and the frame's opening bytes.
  Previously the codec's own error escaped from methods whose every other
  failure is a `ConnectionError` about the endpoint - dialling
  `strands_robots.inference.server` (JSON text frames) with this client answered
  `TypeError: a bytes-like object is required, not 'str'`, naming neither the
  URI nor the port.
