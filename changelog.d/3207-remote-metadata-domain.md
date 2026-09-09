### Fixed

- `RemotePolicy` now holds the metadata a `PolicyServer` advertises in its
  `ready` handshake to the same domain a locally-loaded policy is held to. The
  two chunk counts arrived through a bare `int()` and the two capability flags
  through a bare `bool()`, which was silent rather than lenient: an advertised
  `execution_horizon` of `0` landed behind `Policy.execution_horizon`'s
  `max(1, ...)` floor, so a peer declaring a 16-action chunk was mirrored as
  single-step, `is_chunk_emitting()` answered `False` and `resolve_chunk_length`
  consumed 8 of the 16 actions the peer said it emits, all reported as success;
  `bool("no")` is `True`, so a peer answering `"no"` turned a capability on; and
  a `null` raised `TypeError: int() argument must be a string...` out of the
  middle of a connect, naming neither the field nor the peer.
  `execution_horizon` and `actions_per_step` now share `chunk_count_error`'s
  domain with the constructor parameters they mirror, `requires_images` and
  `supports_rtc` must be JSON booleans, and `provider_name` a string. Every
  field is checked before any is applied, so a refusal leaves the mirror
  untouched, and a connection whose metadata was rejected is discarded on the
  `reset` path as well as on the handshake. A field the handshake omits is still
  not refused, so a peer advertising a subset of the metadata stays usable.
