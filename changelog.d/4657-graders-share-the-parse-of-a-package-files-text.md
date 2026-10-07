### Tests: graders handed a package file's text share its parse too

`tests._package_ast.parse_file` parsed a package file once per process, but
graders whose helpers take the text (so a planted negative can pass a string)
still called `ast.parse(source)` on every package file. They now call
`parse_source`, which returns the shared tree when the text is a package
file's and parses anything else fresh. The fresh-read refusal in
`tests/test_package_sources_are_parsed_once_per_process.py` also matches
`ast.parse(Path(mod.__file__).read_text())` and `ast.parse((ROOT / rel).read_text())`,
which its old pattern let through; the 57 sites spelled that way now use
`parse_file`. On the whole-tree roster (180 modules, 14 workers) package
parses outside the shared cache fell from 37,737 to 9,359 and CPU time from
2,199 s to 2,045 s.
