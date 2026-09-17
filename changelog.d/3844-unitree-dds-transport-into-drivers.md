### Changed: the Unitree DDS transport lives with the drivers that share it

The subscriber layer, the shared init lock, the SDK-absence refusal, the
return-code catalogue and the motion-switcher FSM read moved from
`strands_robots/tools/g1/` to `strands_robots/drivers/unitree/` as `_common`,
`_dds_engine` and `_motion_switcher`. The G1, Go2 and Booster T1 all speak
Unitree IDL over CycloneDDS, so all three drivers were importing upward into the
agent-tool package for their own transport; the verbs now read it back
downward and `strands_robots.tools.g1`'s public names are unchanged.

Seven of the eighteen inversions declared in `scripts/check_import_layers.py`
are gone with it, and `tests/test_import_layers_are_a_dag.py` grades the rule
across runtime, `TYPE_CHECKING` and late imports, so a deferred import cannot
restore the edge behind a function boundary.
