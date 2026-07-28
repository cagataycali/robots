### Fixed: the sim extra declares the IK solver its move_to primitive needs

`move_to` - the Cartesian transport primitive in the MuJoCo backend's
agent-callable action enum - solves inverse kinematics through
`MinkIKBridge`, i.e. through `mink` + `qpsolvers`. No extra declared either
package, so on the advertised install the action was a dead end:

```python
Robot("panda", mode="sim").move_to(position=[0.4, 0.0, 0.4])
# pip install "strands-robots[all]"
# {"status": "error"}  move_to: IK bridge unavailable: ... No module named 'mink'
```

The library documented the gap rather than closing it: the install hint read
`uv pip install 'strands-robots[sim-mujoco]' mink` - an extra plus a package
name no extra provides - and the dev environment installed `mink` by hand, which
is why every IK test passed in CI while the action could not run for a user.

`[sim-mujoco]` now declares `mink` and `qpsolvers`, so `[sim-mujoco]` and
`[all]` can run `move_to`; the dev env drops its hand-installed copies and gets
them through `features = ["all"]` like every other dependency; and the four IK
install hints name only extras, matching the `cosmos3-sim` hint that was already
self-sufficient. `mink` brings its own QP backend (`qpsolvers[daqp]`, the
bridge's first preference), so no additional solver has to be chosen.
