### Fixed: the robot card names a camera that was dropped at connect, with the device's answer

A native driver already recorded each configured camera it could not open under
`presence.camera_failures`, but the dashboard's robot card and detail screen read only
the cameras that did open, so a dropped camera showed nothing at all. Each dropped
camera is now a `wrist: dropped - <reason>` row on both surfaces, with a reconfigure
button that opens the cameras sheet on that camera's row. The reason itself changed
too: a camera that opens but sends no frame used to say "another process may hold
the device"; it now names the mode it asked for, the mode the device answered with,
and any value the backend refused outright.
