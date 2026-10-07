### Fixed

- `Teleoperator('<leader>')` missing-port refusal now names this host's
  serial candidates and the canonical `Teleoperator(name, port=...)` form,
  matching the sibling `Robot(name, mode='real', driver='strands')` refusal
  that already scanned
  (`strands_robots/_serial_discovery.describe_serial_candidates`). Previously
  the user saw only lerobot's raw `__init__() missing 1 required positional
  argument: 'port'` with no scan, no hint, and no copy-pasteable next call.
  The hint fires only when the resolved dataclass declares a `port` field,
  so bus-less teleoperators (`gamepad` / `keyboard` / `phone`) are untouched.
