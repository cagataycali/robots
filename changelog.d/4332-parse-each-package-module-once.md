### Tests: the whole-tree graders parse each package file once per test process

About 180 test modules read and parsed every `strands_robots` source file on
their own, many at module scope, which pytest-xdist repeats on every worker at
collection. They now go through `tests._package_ast.parse_file`, which parses a
package file once per process and hands later callers the same read-only tree
(any path outside the package is still parsed on every call).
`tests/test_package_sources_are_parsed_once_per_process.py` refuses a test that
spells `ast.parse(<path>.read_text(...))` again. On the whole-tree roster plus
the touched modules (271 files, 14 workers) CPU time fell from 3,062 s to
2,602 s.
