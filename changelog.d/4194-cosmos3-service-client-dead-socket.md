### Fixed

- **policies/cosmos3**: after the RoboLab server answers a request with its
  traceback and closes (code 1011), the client drops that connection so the next
  call dials afresh; it kept the dead socket and the next call escaped as a raw
  `websockets.exceptions.ConnectionClosedError`. A connection the peer closes
  mid-session is now reported as the documented `ConnectionError`. (#4194)
