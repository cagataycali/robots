### Fixed: Isaac remove_robot deletes the robot's prim from the stage

`IsaacSimulation.remove_robot` pruned only the in-Python registries and left the
articulation on the USD stage, unlike `remove_camera` (which deletes its prim)
and `remove_object` (which removes through the scene). Because `add_robot`
refuses only a name still in `_robots`, a `remove_robot(name)` then
`add_robot(name)` composed a second reference onto the leftover prim rather than
a fresh one, and only `destroy` ever released it. It now deletes the prim (both
the requested path and the `actual_prim_path` the URDF importer may have
relocated it to) inside the lock, and marks the tensor view stale -- a robot is
an articulation PhysX holds in the view, so deleting it needs a `reset()` before
the next `step`, the same as a dynamic `remove_object`.
