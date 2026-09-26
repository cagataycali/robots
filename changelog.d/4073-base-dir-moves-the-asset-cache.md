### Fixed: STRANDS_BASE_DIR takes the asset cache with it

`STRANDS_BASE_DIR` relocates the directory every piece of strands-robots user
data resolves under - the user registry, the USD and checkpoint caches, the
harness memory - but `get_assets_dir()` resolved against the module-level
`DEFAULT_BASE_DIR` instead of that override. The asset cache is by far the
largest of these directories, so an operator who moved user data onto a volume
with room kept downloading MJCF files and meshes to the home filesystem, with no
diagnostic saying so.

It now resolves through `base_dir_path()`, so `STRANDS_ASSETS_DIR` still moves
the assets alone and `STRANDS_BASE_DIR` moves them with everything else. The
variable also has a row on `docs/reference/configuration.md`, which had never
named it while promising every environment variable the package reads.
