### Fixed: the 29 CodeQL alerts in `strands_robots/mesh`

The `core -> session -> core` import cycle (and its `core -> sensors -> session`
twin) is gone: `mesh_disabled_by_env` now lives in the leaf
`strands_robots.mesh._kill_switch`, `core` re-exports it, and `session` imports
it downward instead of reaching back into `core`. Every bare `except: pass` in
`core`, `sensors`, `iot/provision`, `iot/bootstrap` and `transport/factory`
either logs at debug with the traceback (so a swallowed read or close no longer
hides silently) or says on the line why the exception is expected. The unused
`urllib.request` in `provision.py` was a shadowing bug, not dead code: the
in-function `import urllib.error` rebound `urllib`; it now imports at module
scope. `sensors.put` is a deliberate re-export and is declared in `__all__`.
