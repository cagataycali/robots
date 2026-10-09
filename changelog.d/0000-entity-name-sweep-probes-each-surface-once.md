### Tests: the entity-name sweep drives each creation surface with five probes, not all eighteen values

Every creation site on MuJoCo, Newton and Isaac (`add_object`, `add_camera`,
`add_robot`, and the three name-claiming `patch_scene_mjcf` ops) refuses an
unaddressable name by calling `entity_name_error` (or `camera_name_error`,
which calls it) itself. The full eighteen-value table stays pinned once on that
guard, and each surface gets one probe per way it could drop the guard (`7`,
`0`, `["x"]`, `""`, `"a\x00b"`). The refusal and its "nothing was registered"
half are now one cell per probe. 363 cells become 155 with the same package
lines executed.
