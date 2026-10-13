### Tests: simulation suites drop the cells that ran no package code

43 cells across 9 suites under `tests/simulation/` asserted what `mujoco`
(stepping, rendering, contacts on a hand-written model), `numpy.zeros`,
`concurrent.futures.Future.result`, `logging` formatting, `random.Random`,
`range` or plain `nan`/`issubclass` arithmetic do with a value the package
already refuses. Coverage contexts show each of them executed zero lines of
`strands_robots`, so they go. Cells that read the package's own source or
constants, pin a MuJoCo compile encoding the package reads, or drive package
code stay. The 9 files go from 381 collected cells to 338.
