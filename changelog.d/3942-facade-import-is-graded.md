### Fixed: an import of the package root is graded by the layer rosters

`layer_of` answers `None` for `strands_robots` itself -- it re-exports names
from every layer, so it belongs to none -- and `upward_edges` skips an edge
whose end has no layer. An import of the facade was therefore graded by
neither the runtime nor the deferred inversion roster, and 20 of the 46
lazily re-exported public names resolve into `tools` and 4 into `app`: a
`core` module writing `from strands_robots import Robot` took a dependency on
`app` that both equalities reported as absent. `doctor` and
`dashboard.sim_session` now name the module that defines `Robot`, and
`tests/test_import_layers_are_a_dag.py` refuses the form over all three
import kinds.
