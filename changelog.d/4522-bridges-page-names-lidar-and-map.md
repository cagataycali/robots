### Docs: the bridges page names `lidar` and `map` as LAN-only

`docs/learn/mesh/bridges.md` listed eight LAN-only topic suffixes, but the mesh
also publishes `lidar/summary`, `lidar/state` and `map/info`, and none of them is
in `DEFAULT_BRIDGE_SUFFIXES`. A cloud rule or a `STRANDS_MESH_BRIDGE_TOPICS`
value copied from the page missed both. The page now lists all ten, and one
test reads the page against the code: the bridged list must equal the default
filter and every published suffix must be on one of the two lists.
