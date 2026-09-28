### Changed: the Reachy Mini daemon link moves to `strands_robots.drivers.reachy_transport`

The native Reachy driver resolved its daemon link from
`strands_robots.device_connect.reachy_transport`, so a driver depended on the
package 0.7 removes. The stdlib-only module now lives beside the driver, and
`tests/test_import_layers_are_a_dag.py` pins that no driver names a mesh or
Device Connect module by string, the import form the layer graph cannot see.

Deprecated: `strands_robots.device_connect.reachy_transport` aliases the moved
module with a `DeprecationWarning` and is removed in 0.7 with Device Connect.
Import from `strands_robots.drivers.reachy_transport`.
