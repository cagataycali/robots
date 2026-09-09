### Bug Fixes

- **assets/download, tools/download_assets**: `force` - the flag that decides whether a robot
  whose assets are already present is re-fetched - was read by truthiness on both surfaces,
  and a re-fetch removes the cached directory (`shutil.rmtree`) before fetching it again. So
  `force="false"` (also `"no"`, `"off"`, `"0"`, `1`, `nan`) was indistinguishable from
  `force=True`: measured with one present robot, the cached directory was replaced, a file the
  user kept beside its assets was gone, and the call reported `downloaded: 1` to a caller who
  had spelled the not-re-downloading of it. That directory is the one place in the module a
  user's own files live - every copy filters on read rather than cleaning up afterwards
  precisely so a README or notes beside the assets survive a download. The falsy
  non-booleans (`None`, `0`, `[]`) took the skip branch without being a declared spelling of
  it, and `_needs_download` returned the flag verbatim as its own `bool` verdict. Both
  surfaces now hold `force` to the shared `boolean_flag_error` domain - the shape the two
  other flags in front of an `rmtree` already have - refused before the cache is read, so no
  refusal arrives after the directory it declined to replace is gone.
