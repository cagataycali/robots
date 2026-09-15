### Fixed: a `MotionPlayer` cache read from a file is read as the format this module documents

`MotionPlayer` accepts a reference-motion cache three ways - a `dict`, an `.npz`
from `save_cache_npz`, and a `.pt` that is already a cache - and documents two of
its keys as optional (`control_dt`, `num_frames`). The two file readers stated a
narrower format than that. The `.npz` reader indexed all eight keys itself, so a
file omitting either scalar was refused with NumPy's bare `KeyError:
'control_dt'` and a file short of a channel with `KeyError: 'dof_vel'` - the
first missing channel only, naming neither the player nor the file, where the
`dict` route lists every missing channel. The `.pt` reader decided "this is a
cache" on the presence of `control_dt`, so a cache-shaped `.pt` that omitted it
was sent down the raw-motion path and refused as "Unrecognised raw motion
format", a report about a layout the caller was not using.

Both file routes now hand the cache to the one reader, which is where the
optional keys, the frame-count agreement check and the `control_dt` domain
already live, and a refusal on a file names the file. The cache-shaped `.pt` is
recognised by `body_rot` - the one key no raw layout carries (a packed library
spells its rotations `grs`, a single motion `rigid_body_rot`) - so it no longer
depends on a key the format calls optional, and a raw motion is still resampled.
