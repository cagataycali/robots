### Quality: shipped source may not cross-reference the test tree in either spelling

The guard that refuses a test-file citation in `strands_robots` matched only a
`.py` filename, so the same dead end written as a dotted module path - a
`:mod:`/`:data:` role such as `tests.simulation.mujoco.test_tool_spec.X` - read
as a checkable cross-reference and passed. It is not checkable: the wheel ships
`strands_robots` alone, so `import tests.simulation` raises `ModuleNotFoundError`
for anyone who installed the distribution, and renaming the test moves the target
just as silently as the filename form does.

Both spellings are now refused, with the narrowing pinned: a slash path into
another project's repository (an upstream `tests/authentication.rs`) names a file
outside this distribution and stays ordinary prose.
