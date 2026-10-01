### Fixed: an Isaac camera read after an action shows the action, not the moment before it

The RTX render product delivers a tick behind, and a camera read after
`send_action` or `step` returned the frame from before the action: go2 folded
flat under `send_action(n_substeps=40)` while `render` - twice - still showed it
standing. With one camera, `get_observation` refreshed nothing; with several,
one render-only tick, which is still one short.

`render` and `get_observation` now give the renderer two render-only ticks
(`SimulationApp.update()`, no physics step) whenever physics has stepped since
the last camera read; repeated reads between steps cost nothing, and `reset()`
clears the marker.
