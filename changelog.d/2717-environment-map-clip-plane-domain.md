### Fixed
- **rendering**: `render_environment_map` checks `znear` / `zfar` against the shared
  `positive_finite_number_error` domain and requires `znear < zfar`, before any cube
  face is rendered. An inverted or equal pair - `znear=10.0` with `zfar=1.0` - was
  accepted and returned a full-size, entirely black map reporting success, after
  paying all six background renders (GPU-bound for a `GsplatBackground`); the only
  refusal the caller ever saw came from `derive_key_light`, blamed the scene ("the map
  is black above the horizon"), and advised a search flag that fails the same way, so
  neither clip plane was named anywhere. The planes cannot be left to the background:
  with the same arguments `PanoramaBackground` ignores them and returns its usual map
  while `GsplatBackground` forwards them to `gsplat.rasterization` and culls every
  gaussian. An ordered pair that happens to frame nothing stays accepted, matching the
  resolution knobs, which check the domain rather than the quality.
