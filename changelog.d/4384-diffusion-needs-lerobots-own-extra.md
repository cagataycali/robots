### Docs: a diffusion checkpoint needs lerobot's own `[diffusion]` extra

The lerobot_local page listed diffusion among the types `strands-robots[lerobot]` runs; that extra installs `lerobot[feetech,dataset]` and a `DiffusionPolicy` refuses without `diffusers`. The install fence now names `pip install 'lerobot[diffusion]'` (and `lerobot[pi]` for pi0), and a test keeps the fence and pyproject in agreement. (#4196)
