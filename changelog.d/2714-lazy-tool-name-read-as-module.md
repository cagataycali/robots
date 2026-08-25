### Fixed

- **tests**: two tests no longer bind a lazily-mapped tool name off
  `strands_robots.tools` and then read the result as a module. The package maps each
  tool name to the `@tool` object inside the submodule of the *same* name, so
  `from strands_robots.tools import use_rosbridge` has two possible resolutions and
  which one wins is decided by whichever import ran first anywhere in the process:
  CPython imports the submodule only when the attribute is *absent*, and here the
  lookup triggers the package `__getattr__`, which succeeds. A `DecoratedFunctionTool`
  carries no `__file__`, no `__spec__` and none of the module's private names, so both
  sites passed on an unrelated import rather than on the behaviour they verify -
  `test_rosbridge_transport_port_limit.py` on a neighbouring import staying above it
  (swapping those two alphabetically-adjacent lines is a collection error that loses
  all 24 tests in the file), and `test_output_path_is_confined_to_its_directory.py` on
  two earlier tests in the same file, so a `-k` filter selecting that test alone
  failed with `AttributeError`. Both now import the submodule directly. A new
  package-wide guard refuses the shape over this project's own sources and derives
  both halves of the rule - which mapped names are ambiguous, and which attributes
  only a module can answer, asked of the tool object rather than listed - so a tool
  added to the mapping and a call site added to any scanned tree are graded on
  arrival. No shipped behaviour changes.
