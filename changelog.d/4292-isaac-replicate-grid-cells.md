### Fixed: Isaac replicate gives every environment its own grid cell and keeps each clone's pose

`replicate(num_envs=N)` handed Isaac's `GridCloner` the N-1 clone paths, and
GridCloner lays out a grid for the paths it is given, centred on the origin,
while the source scene stayed at the origin as env_0. With `num_envs=4,
spacing=1.5` the environments sat at x = 0 (source), 1.5, 0, -1.5: env_2 was on
top of env_0, to 1e-7 m, and the inter-environment collision filter hid it. The
cloner also moved each clone root to its cell, so the source's own pose was
lost: a cube authored at (0.3, 0.3, 0.02) was cloned to (x, y, 0), inside the
ground.

Environments are now laid out row-major on a square grid with env_0 at the
origin, and each clone is placed at its source's world pose plus its
environment's offset. The offsets are in the result as `env_origins`.
