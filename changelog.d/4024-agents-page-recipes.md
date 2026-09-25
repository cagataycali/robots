### Docs: the agents page's recipes import under its own install and build the size they state

`docs/agents.md` told a reader to `from strands_robots import gr00t_inference,
pose_tool` under its own `strands-robots[sim-mujoco]` install line. `pose_tool`
requires `pyserial` to import and no extra of this project declares it (it
arrives inside `lerobot[feetech]`), so following the page top to bottom ended in
`ImportError: cannot import name 'pose_tool'`. The page now states that
dependency, as `docs/hardware/tools.md` does.

Its "Common patterns" table also mapped "Add a 5cm red cube" to
`size=[0.025]*3`. `size` is the full extent, so that compiled a 2.50 cm cube -
the half-extents MuJoCo stores for a 5 cm one. The row now reads `[0.05]*3`.
