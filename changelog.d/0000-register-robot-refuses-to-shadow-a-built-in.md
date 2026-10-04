### Fixed: `register_robot` refuses a built-in robot's name unless `overwrite=True`

Registering a name the package already ships (`so101`, `g1`, `panda`, ...)
used to succeed and replace the curated entry - its joint labels, aliases and
hardware block - for every later `get_robot`, `Robot(name)` and
`sim.add_robot(name)`. The only signal was an INFO log, below the default
WARNING floor. The same call for a name already in the user overlay raised.
Both collisions now raise `ValueError` the same way; the built-in message names
the description, joint count and aliases the registration would hide.
`overwrite=True` still shadows the built-in, and `unregister_robot` restores it.
