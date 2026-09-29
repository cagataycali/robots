### Fixed

- **policies/cosmos3**: after the RoboLab server answers a request with its
  traceback and closes (code 1011), the client drops that connection so the next
  call dials afresh; it kept the dead socket and the next call escaped as a raw
  `websockets.exceptions.ConnectionClosedError`. A connection the peer closes
  mid-session is now reported as the documented `ConnectionError`. The dial
  passes `legacy=True` where `websockets` declares it, silencing the
  `connect() must be used as a context manager` deprecation warning that 17.1
  emits on every connection. (#4194)
