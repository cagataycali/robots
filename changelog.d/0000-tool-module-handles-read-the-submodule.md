### Fixed: a test's tool-module handle no longer turns into the tool

`import strands_robots.tools.<name> as mod` reads `getattr(strands_robots.tools,
"<name>")`, not the submodule, so once anything in the process has cached the
tool in that slot - `from strands_robots.tools import *`, a `getattr` over
`__all__`, an agent loading the package - `mod` is the `DecoratedFunctionTool`
and every `mod.<attribute>` read raises `AttributeError`. 104 such handles in 78
test modules now use `importlib.import_module(...)`, which always returns the
module. With every tool cached first, those modules went from 37 collection
errors to the 3,651 passes an unprimed run gets. The import-spelling rule in
`tests/tools/test_lazy_tool_name_imports_are_unambiguous.py` now flags the alias
form beside `from strands_robots.tools import <name>`.
