### Fixed

- **tests, examples**: a lazily exported tool name is read off its submodule, never off the
  `strands_robots.tools` package. The package maps each of its 21 exported names to the `@tool`
  object inside the submodule of the *same* name and caches the result in the package `__dict__`,
  and 16 of those names are also submodules - so `from strands_robots.tools import <name>` binds
  either the tool or the module, decided by what the process imported first: cold it is the tool,
  and after any import of the submodule it is the module. Four sites read that spelling, in both
  failure directions, and each passed in the selection it was written against:
  `test_rosbridge_transport_port_limit` and `test_output_path_is_confined_to_its_directory` used the
  result as a module (`'DecoratedFunctionTool' object has no attribute '__file__'` /
  `... no attribute 'pose_tool'`), while `examples/08_discover_lerobot.py` and
  `examples/libero/run_isaac_agent.py` used it as a tool (`module ... has no attribute
  '__wrapped__'`). Running the pose-tool envelope test on its own reproduced the second test's
  failure on `main` with no edit. That read is also the only spelling that writes the *tool* into
  the slot, which is what makes the module-alias form this tree uses widely as a monkeypatch target
  (`import strands_robots.tools.use_rosbridge as rb_mod`) resolve to the tool instead, so
  `rb_mod.roslibpy` raises `AttributeError` naming the tool class rather than an import order -
  eliminating the ambiguous read tree-wide is what keeps that idiom sound. Both remedies read the
  submodule and are unaffected by import order: the module-alias form where the module object is
  wanted, `from strands_robots.tools.<name> import <name>` where the tool is wanted. A derived guard
  bans the ambiguous read across every top-level area that ships Python, deliberately leaving a name
  with no matching submodule (the `episode_judge` helpers) in scope for nobody, and AGENTS.md
  records the convention.
