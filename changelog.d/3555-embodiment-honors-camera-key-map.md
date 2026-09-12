### Fixed: `camera_key_map` routes cameras on the declarative `embodiment` path too

The `lerobot_local` path taken when an `embodiment` is declared merged only
`obs_rename_override` over the embodiment's `obs_rename`, so a scene whose
cameras are named for the scene rather than for the embodiment (`cam_top` vs
`front`) was refused by the pre-download check - which named
`obs_rename_override` as the remedy for a caller who had already bound the
cameras with `camera_key_map` - and, past that check, reached the model with no
image input at all: the rename map still named the embodiment's absent sources,
so the frames were neither renamed onto the model's image features nor
recognized as camera frames (they stayed HWC `uint8`). `camera_key_map` is now
routed over the embodiment's declared renames, replacing the source key for each
image feature it claims, before `obs_rename_override` (which stays last because
a falsy value there is the only way to DROP a rename). This matches the three
other camera routers, which have always resolved `camera_key_map` first.
