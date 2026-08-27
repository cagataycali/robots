### Added: Pollen Microduck, the 14-DOF open biped

`Robot("microduck")` now resolves. Pollen's Microduck is a 25 cm biped driven by
fourteen Dynamixel XL330 servos, and the MJCF it ships in
`pollen-robotics/microduck_rl` (Apache-2.0) auto-downloads through the existing
github source mechanism, so the entry needs no new machinery.

The declared `joints` figure is 15 rather than the robot's fourteen servos,
because the registry counts MuJoCo's `njnt` with the floating base included.
That is the convention `docs/robots/arms.md` states ("Joint counts include any
free joints"), the one `asimov_v0` follows for the same
one-free-plus-fourteen-hinge shape, and the one twelve of the sixteen humanoids
whose asset compiles already use - none of them declares its actuator count. The
hardware figure lives in the description, as it does for `op3` (21 against
"20-DOF") and `unitree_h1` (20 against "19-DOF").

No home pose is copied into `robots.json`. No entry declares one, `add_robot`
reaches a pose by name from the source model (`keyframe="STAND"`), and upstream
has already retuned this pose once - the superseded `STAND` is still commented
out beside it. A second copy would drift silently, so the shipped keyframe is
asserted against the documented values instead of duplicated beside them.

The joint order the entry documents is load-bearing rather than decorative,
because upstream ships a variant that does not share it:

```
                                     njnt  nu  qpos of neck_pitch
robot_allcollisions.xml (declared)     15  14  12
robot_allcollisions_rollers.xml        19  14  14
```

The rollers variant inserts two passive wheel joints after `left_ankle`, which
moves nine of the fourteen actuated joints to a different `qpos` index - so a
consumer reading joint positions as a flat `qpos[7:21]` slice reads different
joints there. The actuator order is identical across the variants, so a policy
writing `ctrl` is unaffected; only a position read is. The entry therefore names
the fourteen-hinge model, and a test pins that it keeps doing so.

Two test layers, because the oracle is not available everywhere. The entry's
fields are graded from `robots.json` alone, so they hold on an install with no
MuJoCo and no assets; the compiled shape - joint order, actuator pairing, the
keyframe - is graded where an asset can be present. Neither layer downloads
anything, so a host with no network skips rather than fails.

Not included: a dedicated docs page. The library ships no Microduck policy
provider, so a page describing upstream's ONNX policy family would advertise a
capability that is not here; the robot is documented as a catalog row and a
render beside the other bipeds.
