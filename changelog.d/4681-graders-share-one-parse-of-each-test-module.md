### Tests: graders that walk `tests/` share one parse of each test module

`tests._package_ast.parse_file` and `parse_source` shared their tree only for
package files. About 30 graders walk the 2,200 files of `tests/` and each
parsed every one of them again, some twice. Files under `tests/` and
`tests_integ/` now share one tree per process too. Each tree is parsed in one
pass the first time any of its files is asked for, then one collection and
`gc.freeze()` move the trees out of the collector's reach, so holding them
does not make every later collection walk them. On the 60 costliest of those
grader modules, on 2 workers pinned to 2 cores, CPU time fell from 768 s to
601 s and wall time from 407 s to 326 s; peak memory per worker rose by about
0.6-0.8 GB.
