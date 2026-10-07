### Fixed: the "no world" error names `Robot(name, mode="sim")`, the same on every backend

After `robot.cleanup()`, re-running a cell answered `No world. Call create_world (or load_scene) first.` - two engine calls the quickstart never shows. Every backend (MuJoCo, Newton, mjlab, Isaac Sim and the shared recording mixin) now returns one message from `strands_robots.simulation.base`: `No world: none was built yet, or cleanup() tore it down. Construct a fresh Robot(name, mode="sim"), or call create_world / load_scene.` Before, 40 call sites spelled it six different ways.
