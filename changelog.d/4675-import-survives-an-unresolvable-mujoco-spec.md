### Fixed: `import strands_robots` survives a MuJoCo spec lookup that raises

`import strands_robots` no longer crashes when `importlib.util.find_spec("mujoco")` raises - a `mujoco` stub with `__spec__` of `None` left by a mock or a frozen app, or a meta-path finder that refuses the name. The MuJoCo GL hint is skipped instead, as it already is when the hint itself cannot be set.
