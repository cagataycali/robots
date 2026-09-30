### Fixed: the dashboard no longer imports loop pacing through its deprecated path

The dashboard's USB auto-spawn poller and its record worker still imported
`Ticker` from `strands_robots.mesh.pacing`, the shim #4114 left behind, so
the first loop to start raised that shim's `DeprecationWarning`, and both
loops would have failed with `ImportError` once the shim is removed in 0.7. They
now import `strands_robots._pacing`. `tests/test_import_layers_are_a_dag.py` pins that no
package module imports a location kept only for one minor: a module whose body
warns `DeprecationWarning`, or a ROS name or module stem the mesh re-exports
from `strands_robots.drivers.ros`.
