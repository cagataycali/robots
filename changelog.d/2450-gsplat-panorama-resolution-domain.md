### Fixed
- **rendering**: `bake_gsplat_panorama` checks `face_size` / `equi_w` / `equi_h`
  against the same shared `positive_whole_number_error` domain its renderer uses,
  and normalizes them with `int()`, before it composes its default output path or
  loads the splats. That default path is the cache key, so `face_size=640.0` -
  which the domain accepts - previously spelled `..._f640.0.jpg` and re-baked the
  warm `..._f640.jpg` beside it. And an unusable resolution was refused only after
  the splat load: without the `sim-gs` extra installed the load answered first,
  so `equi_w=0` was reported as a missing `torch` and advised an install that
  leaves the resolution exactly as unusable.
