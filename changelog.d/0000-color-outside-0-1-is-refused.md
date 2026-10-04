### Fixed: a colour channel outside 0..1 is refused instead of written verbatim

Every backend documents `add_object(color=...)` channels in 0..1, but
`coerce_rgba` only checked count and finiteness, so a web-style `[255, 0, 0]`,
a negative channel or `1e9` returned `success` and landed raw in the model's
`geom_rgba` (and in `set_geom_properties` and `patch_scene` `rgba`). The shared
helper now refuses any channel outside 0..1 on all backends, and a 0..255
colour is told to divide by 255.
