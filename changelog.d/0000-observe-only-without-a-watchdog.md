### Fixed: a hardware device mounted without a watchdog is served observe-only

`Mesh.mount(robot, watchdog_ms=None)` turns off the watchdog: whoever moves the device no longer has to stay
present. On a simulated body that is harmless, but a real arm could be driven by a mover that then vanished, with
nothing left to recall it to a safe pose. A real body mounted this way is now served observe-only: `observe` and
`describe` answer as before, while `act` and `hold` are refused by name over the mesh, naming `watchdog_ms=` as the
way to make it movable. Nothing is written to the bus. Simulated bodies and real bodies mounted with a watchdog are
unchanged.
