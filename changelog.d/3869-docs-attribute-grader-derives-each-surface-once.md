### Fixed: the documented-attribute grader derives each surface once per pass

`tests/test_docs_robot_attribute_reads_resolve.py` resolves every documented
`Robot(...)` read to the surface of the type the factory returns, and it derived
that surface afresh for every read: `dir()` plus an `inspect.getsource` and
`ast.parse` over each class in the MRO. `Simulation` carries a 14-class MRO that
costs ~0.5 s to walk, and the 48 sim-mode reads in the docs paid it 48 times -
21 s of a 25 s cell, the largest single-cell walk left in the suite per #3869.
`_instance_surface` and `_factory_bound_names` are now memoised and return
`frozenset`s, so a pass fetches each class source once; the sim cell runs in
0.6 s and the hardware cell in 0.4 s with every assertion unchanged. A new cell
pins that a second pass over the corpus fetches no class source.
