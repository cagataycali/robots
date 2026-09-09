### Bug Fixes

- **device_connect/reachy_mini**: `enableMotors` / `disableMotors` read the
  comma-separated `motor_ids` selector by truthiness, so a non-empty selector
  that names no motor - `","`, `" "`, `",,"` - parsed to `[]`, was coalesced to
  the same `None` the documented `""` default resolves to, and torqued every
  motor while the reply echoed the caller's own selector as the set acted on.
  Such a selector is now refused by value before anything reaches the hardware
  link; `""` still selects every motor and a well-formed list is unchanged.
