### Fixed: Isaac `send_action` is main-thread affine, like `step`

`send_action` stepped the world directly on whatever thread called it, so a
worker thread's `send_action` (the path an agent's `run_policy` reaches through
`asyncio.to_thread`) blocked forever inside PhysX's native `step`. It now routes
through the same main-thread marshal as `step` and `reset`: it runs inline on the
`SimulationApp`-owning thread, hops onto the pump when `run_pump_forever` is
engaged from a worker, and refuses with a `RuntimeError` naming the recipe when a
worker calls with no pump. Substeps are batched under `_STEPS_PER_BATCH` so the
lock is released between batches. The `pump()` docstring now describes this model,
and `pump` no longer refreshes two caches (`_joint_cache`, `_frame_cache`) that no
production code read; the idle-preview RTX readback that fed them
(`_grab_frame`/`_resize_rgb`/`_cam_out_size`) is removed with them.
