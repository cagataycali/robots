### Fixed

- **policies/cosmos3**: the in-process backend passes the load dtype under the
  keyword the installed `diffusers` declares (`dtype` from 0.40, `torch_dtype`
  on the 0.39 floor), so a load no longer emits `FutureWarning: torch_dtype is
  deprecated`. The docs page names the diffusers backend's raw action, the sim
  route through `decode_cosmos_chunk_to_targets` + `MinkIKBridge`, the
  client-side camera rule, the `openarm` domain gap on `diffusers` 0.40 and the
  server's sampling defaults. (#4194)
