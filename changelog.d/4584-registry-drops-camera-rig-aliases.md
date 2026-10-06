### Removed: the `so100_4cam`, `so100_dualcam`, `so101_dualcam` and `so101_tricam` aliases

They resolved to the bare `so100` / `so101` arm, which declares no cameras, so
`Robot("so100_4cam")` built a scene with only MuJoCo's free `default` camera and
said nothing. They were GR00T data-config names, and GR00T has left the package.
Use `Robot("so100")` / `Robot("so101")` (or `so100_follower` / `so101_follower`)
and add cameras with `add_camera(...)` in sim or `cameras={...}` in
`mode="real"`. The old names now raise `Unknown robot ... Did you mean: so100?`.
A registry test refuses any alias whose name advertises a camera rig.
