### Fixed: the Reachy Mini page states the vocabulary its driver dispatches

`docs/hardware/reachy-mini.md` said the native tool exposed only `sensors`,
`status` and `stop`, and that camera capture, audio playback, volume and
pixel-directed look were not implemented; `ReachyDriver` declares twenty-four
actions, all four among them. The page now carries the whole action table,
capture, pixel look-at and the acknowledged-antenna mode, moved off the
`robots/humanoids.md` catalog page where that reference had accumulated below the
See also block, and a grader reads the roster from the driver's own `tool_spec`.
