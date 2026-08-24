### Fixed: an MJCF's bodies are read from the whole model, not just the top file

MuJoCo splices `<include file=...>` textually and merges every `<worldbody>` the
spliced model carries, the same way it treats `<compiler>` and `<asset>` as
model-global. Both loaders in `strands_robots.simulation.isaac.loaders` still
answered that question from the top file's own `<worldbody>`, so whether a body
was visible depended on which file declared it.

That failed in two directions. A model whose `<worldbody>` lives entirely in a
fragment read as having none, so `load_mjcf` refused a model MuJoCo compiles -
`aloha/scene.xml` is an `<include>` plus a table, and it was rejected as a
"phantom robot" while `aloha/aloha.xml` returned all 21 bodies. A model keeping
some bodies locally and including the rest read only the ones it could see and
silently dropped the others, their subtrees and their joints:
`franka_emika_panda/mjx_single_cube.xml` returned 2 bodies and zero joints for a
model with 13 and 10, so the whole Panda arm was absent from a
`ProceduralRobot` the caller was told it received.

Both loaders now read the bodies through the module's existing splice, in
document order (which is the order MuJoCo assigns body indices in), and keep "no
`<worldbody>` anywhere" distinguishable from "a `<worldbody>` with no bodies" so
both refusals still report their own cause. Measured over the 227 downloaded
registry MJCFs: 75 refusals of models MuJoCo compiles are fixed, 12 silent
body/joint drops are fixed, and 3 scenes now report the articulation refusal
their own included model file already reported. The 106 LIBERO asset scenes -
the only in-package consumer of the scene loader, and they use no `<include>` -
return byte-identical object lists.

An `<include>` nested inside a `<worldbody>` stays out of scope: no shipped
registry `<worldbody>` uses one, and it is a question about splicing inside a
`<worldbody>` rather than about which `<worldbody>` elements a model has.
