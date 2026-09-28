### Fixed: the Isaac backend's sim clock now tracks the physics it drives

`IsaacSimulation` credited a constant `config.physics_dt` per `step()` while the
`World` integrated a different amount, so `_sim_time` (what `get_state`,
recordings and elapsed-time reports read) and `physics_timestep()` (what
`PolicyRunner` sizes its substeps from) drifted from the physics in three cases,
measured on Isaac Sim 6.0.1 against `World.current_time`:

- a rendering `step()` folded Kit's `app.update()` into `World.step(render=True)`,
  integrating a whole `rendering_dt` (four `physics_dt` substeps at the defaults)
  per tick while the clock counted one -- an `rtx_realtime` scene under-reported
  its elapsed time ~4x;
- a `create_world(timestep=)` override is honoured by `World` but never written
  to `config.physics_dt`, so the accumulator and `physics_timestep()`
  over-reported the timestep;
- the idle render pump (`_converge_render`) advanced physics every tick while
  touching neither `_sim_time` nor `_step_count`.

`_sim_time` is now read back from `World.current_time` after each tick, and a
rendering `step()` steps physics exactly once (`render=False`) and refreshes the
frame with a separate `World.render()`, so one `step()` is one `physics_dt` in
every render mode and `PolicyRunner`'s substep count stays correct without
change. `physics_timestep()` reports the dt the `World` is actually integrating,
and `_converge_render` renders without advancing physics.
