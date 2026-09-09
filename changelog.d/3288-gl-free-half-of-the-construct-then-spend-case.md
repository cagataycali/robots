### Tests: the GL-free half of the engine default-resolution case keeps running

`test_every_engine_that_constructs_can_be_rendered_and_observed` was gated whole
on the shared MuJoCo GL probe, and one of its assertions needs no OpenGL context:
`create_world` copies the constructor's `default_width` onto the free camera's
`SimCamera` entry, and that entry reads back the same value on a host with none.
That copy is the second way into the field `add_camera` guards - the surface the
domain was added for - so gating it together with the render left it unchecked
exactly where `tests/simulation/mujoco/_gl_probe` records that GL-backed
contracts used to go unverified. Measured by breaking only that propagation and
counting the cells that fire: under `MUJOCO_GL=egl` the gated case reports 1 and
the split reports 2; under MuJoCo's documented `MUJOCO_GL=disable` the gated case
reports 0 and the split reports 1. The case is now two - the value the
constructor stored, which runs wherever the suite runs, and `render` /
`get_observation`, which stay behind the probe because `MUJOCO_GL=disable`
reports `{"status": "error"}` for the first and omits the image key for the
second, so reading it raises `KeyError: 'default'`.
