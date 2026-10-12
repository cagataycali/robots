### Tests: policy, mesh and driver suites drop the cells that ran no package code

64 cells across 21 suites under `tests/policies/`, `tests/mesh/` and
`tests/drivers/` asserted what `threading.Event.wait`, `socket.settimeout`,
`json`, `requests`, `botocore`, `numpy`, `deque`/`range` or plain Python
comparisons do with a value the package already refuses (`not nan < 1`,
`Event().wait(True)` waiting one second). Coverage contexts show each of them
executed zero lines of `strands_robots`, so they go. Cells that read the
package's own source or constants, check a vendor wire value, or drive package
code stay. The 21 files go from 763 collected cells to 699.
