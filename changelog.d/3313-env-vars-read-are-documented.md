### Docs: every `STRANDS_*` environment variable the package reads has a row in the README

Fifteen variables the package honoured appeared on no page under `README.md`
or `docs/`. Seven are read with `os.getenv` directly:
`STRANDS_MESH_CAMERA_S3_BUCKET` and `_PREFIX` (the two that turn the camera S3
offload on - the TTL that only matters once it is on was documented),
`STRANDS_GR00T_REPO_URL` and `_TAG` (the clone source `build_image` fails
closed on - its allowlist was documented without the variable it constrains),
`STRANDS_MESH_BRIDGE_DEDUP_STRICT`, `STRANDS_MESH_FILTER_INTERFACES` and
`STRANDS_ROBOTS_VERBOSE_MUJOCO`. Eight are read through the package's own
resolvers (`_int_env`, `_float_env`, `_bool_env`, `_env_int`): the six mesh
transport bounds `STRANDS_MESH_MAX_CMD_BYTES` / `_MAX_CAMERA_BYTES` /
`_MAX_SAFETY_BYTES` / `_CMD_RATE_HZ` / `_SAFETY_RATE_HZ` / `_MAX_SESSIONS`, the
camera privacy switch `STRANDS_MESH_CAMERA_DISABLED`, and
`STRANDS_ISAAC_CAMERA_WARMUP_STEPS`. Each gets a row beside the sibling it
belongs with, worded from its read site.

`tests/test_env_vars_the_package_reads_are_documented.py` derives the
population from the package by AST - direct reads and reads through any
function that reads the environment via one of its parameters, found to a
fixed point and followed through import aliases - and the documented set from
the pages, so a variable added later is graded on arrival rather than by
whoever remembers the README rule (#3313).
