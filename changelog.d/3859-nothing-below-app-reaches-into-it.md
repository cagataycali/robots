### Changed: `RecordingFrameError` is a core module, and the teleop mixin is declared where its hosts are

`RecordingFrameError` moves from `strands_robots/dataset_recorder.py` to a new
`strands_robots/recording_errors.py`. It is raised in the `app` layer by
`DatasetRecorder.add_frame` and caught one layer down by the rollout drivers in
`strands_robots.simulation.policy_runner`, which imported the recorder module -
numpy and LeRobot version probing included - to name one exception. The name
stays importable from `strands_robots.dataset_recorder`, which raises it.

`teleop_mixin` is declared a `drivers|mesh` member in
`scripts/check_import_layers.py` rather than an `app` one, with no change to the
module: it reads `strands_robots.utils` alone at module scope and is mixed into
three hosts in three layers - `hardware_robot`, the MuJoCo `Simulation` and
`device_connect.sim_driver` - so it belongs under the lowest of them rather than
with the first host that needed it.

Together those remove all three inversions that pointed into `app`, 7 declared
upward runtime edges to 4, leaving one family: the four mesh robots that call an
`@tool` entry point. `tests/test_import_layers_are_a_dag.py` gains the property
as a pin - no runtime edge targets `app` - plus one parametrised row per
placement stating the layer, the imports that justify it, the caller layers that
need it, and the single late read above the layer (`teleoperator`, deferred
because it imports lerobot) by name rather than in general.
