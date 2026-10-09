### Tests: a grader that is handed a file's text parses it through the shared tree

About a hundred test modules read a package or test file and parsed the text
with `ast.parse` themselves, through a helper parameter or a local the old
one-line rule could not see, so each parsed again what
`tests._package_ast` already holds. They now call `parse_source`, which hands
back the shared tree for a graded file's text and parses any other text
afresh, and `tests/test_package_sources_are_parsed_once_per_process.py` reads
the three spellings (inline read, a name bound from a read, a helper's
parameter) from the AST. The 107 touched files take 26-29 s less CPU on two
workers.
