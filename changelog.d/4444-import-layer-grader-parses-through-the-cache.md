### Tests: the deprecated-location import check parses through the shared cache

`tests/test_import_layers_are_a_dag.py` gained a check that reads every package
module for imports of a moved location, spelled `ast.parse(path.read_text(...))`.
It landed alongside the grader that refuses that spelling, so `main` failed
`test_no_test_module_parses_a_source_file_afresh`. The three reads now go through
`tests._package_ast.parse_file`.
