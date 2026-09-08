### Docs: every `STRANDS_*` environment variable the package reads has a row in the README

Seven variables the package honoured appeared on no page under `README.md` or
`docs/`: `STRANDS_MESH_CAMERA_S3_BUCKET` and `_PREFIX` (the two that turn the
camera S3 offload on - the TTL that only matters once it is on was documented),
`STRANDS_GR00T_REPO_URL` and `_TAG` (the clone source `build_image` fails
closed on - its allowlist was documented without the variable it constrains),
`STRANDS_MESH_BRIDGE_DEDUP_STRICT`, `STRANDS_MESH_FILTER_INTERFACES` and
`STRANDS_ROBOTS_VERBOSE_MUJOCO`. Each gets a row beside the sibling it belongs
with, worded from its read site.

`tests/test_env_vars_the_package_reads_are_documented.py` derives the
population from the package by AST and the documented set from the pages, so
a variable added later is graded on arrival rather than by whoever remembers
the README rule (#3313).
