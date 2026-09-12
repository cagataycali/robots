### Fixed: an explicit camera_key_map outranks another camera's name match on the preprocessor path

`camera_key_map` is routing precedence rung 1 - "an explicit mapping wins for
any name it lists" - and there are two routers that have to implement it.
`_resolve_camera_targets` builds the batch when the checkpoint ships no
preprocessor and resolves the map in its own first pass. `_to_lerobot_observation`
remaps the observation when the checkpoint does ship one, which is the case for
every locally trained ACT/diffusion/SmolVLA checkpoint, and it resolved the map
per-camera inside the same loop as the exact-name match. So the precedence
depended on observation order: a camera iterated earlier that matched a declared
key by name claimed the slot, and the mapped camera was then dropped for having
no free slot left, with nothing reporting the discarded entry.

Driven on one three-camera sim observation (`default`, `top`, `wrist`) with
`camera_key_map={"wrist": "observation.images.top"}`, the two routers disagreed:

    _resolve_camera_targets : observation.images.top <- wrist    (as asked)
    _to_lerobot_observation : observation.images.top <- top      (silently)

The map is now applied over the whole observation before any exact-name match
runs, mirroring the first pass `_resolve_camera_targets` already does, so the two
routers answer the same input the same way - whether a checkpoint ships a
preprocessor is not something a camera binding may depend on. The map's own
validation is unchanged: an entry naming an image key the policy does not
declare still raises on both paths, and a camera the map does not name still
binds by exact name and then through the documented positional fallback.
