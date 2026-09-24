### Fixed: the MuJoCo-GS example points only at modules and actions that exist

`examples/mujoco_gs/README.md` still ran `python -m
examples.mujoco_gs.app_groot_libero` and `python -m
examples.mujoco_gs.libero_groot` on three lines, under a "verified recipe" for a
GR00T + LIBERO demo whose scripts left with the vendored LIBERO adapter, and the
scene block of the demo's agent system prompt closed with "then call
`hybrid_render`" - a name the `Simulation` tool does not publish, so a model that
followed it spent its turn on `Unknown action: hybrid_render`. The section is
gone, the prompt names `render`, and two sweeps now grade a shipped example's
`python -m` commands and prompt actions against the tree.
