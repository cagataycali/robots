### Fixed: a robot whose model cannot compile no longer bricks the scene

`add_robot` attaches the robot's subtree into the live `MjSpec` before the
recompile that validates the result. When that recompile was refused - a robot
model referencing a mesh file that cannot be opened attaches fine and is only
rejected at compile time - the subtree was left in the spec. Every later scene
mutation recompiled that same broken spec and failed too, so one unloadable
robot bricked the whole world:

```python
sim.add_robot(name="badbot", urdf_path="model-with-a-missing-mesh.xml")
# error: Failed to inject robot 'badbot' into scene.   (expected)

sim.add_object(name="marker", shape="sphere", position=[0.0, 0.5, 0.1])
# before: error: Failed to inject 'marker': spec recompile refused.
# after:  success
sim.add_camera(name="look", position=[1.5, -1.5, 1.0], target=[0, 0, 0.2])
# before: error: Failed to inject camera 'look': spec recompile refused.
# after:  success
sim.add_robot(name="panda")
# before: error: Failed to inject robot 'panda' into scene.
# after:  success
```

Nothing was wrong with any of those calls, and each failed retry left another
orphan subtree behind, so the spec drifted further from the scene `world` still
described. The attach is now rolled back out before the failure is reported, so
a refused robot costs exactly the add that was refused - matching
`add_object` and `add_camera`, which have always rolled their spec mutation
back.

The rollback reinstalls a snapshot of the spec taken before the attach. It
cannot rebuild the scene from the registered objects/cameras/robots instead,
because the live spec can carry mutations that registry never records - weld
equalities from `attach_bodies`, actuators from a robot actuated for IK, bodies
authored by `patch_scene_mjcf`, whole scenes from `replace_scene_mjcf` - and
dropping those would turn a correctly refused add into corruption of a scene
that was healthy before it.

The snapshot is a spec copy rather than a `spec.to_xml()` round trip, which
fixes the same rollback where it already existed: for a scene holding an
attached robot whose meshes load from files, the emitted MJCF loses the asset
search paths those references were resolved against and re-declares the model's
keyframes, so restoring it put an uncompilable spec back. A refused
`patch_scene_mjcf` batch (or a failed `actuate_robot_in_scene`) therefore left
the next unrelated mutation failing with `Error opening file 'link2.stl'`:

```python
sim.add_robot(name="panda")                       # meshes load from files
sim.patch_scene_mjcf(ops=[..., <an op that fails>])
# error: patch op #2 failed: ...                  (expected)

sim.add_object(name="marker", shape="sphere", position=[0.0, 0.5, 0.1])
# before: error: Failed to inject 'marker': spec recompile refused.
# after:  success
```
