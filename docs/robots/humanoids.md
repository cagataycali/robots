---
description: Full-body humanoids and expressive desktop robots.
---

# Humanoids

Full-body humanoids and expressive desktop robots.

```python
from strands_robots import Robot
sim = Robot("unitree_g1")       # Unitree G1
sim = Robot("unitree_h1")       # Unitree H1
sim = Robot("apollo")           # Apptronik Apollo
sim = Robot("reachy_mini")      # Pollen Reachy Mini (expressive)
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:humanoid, expressive}}

## Real hardware

Every humanoid here with a native driver has its bring-up on its own page - the
port to pass, which joints a host may command, and which verbs the onboard
controller keeps for itself:

- [Booster T1 over the vendor SDK](../hardware/booster-t1.md)
- [Unitree G1 over CycloneDDS](../hardware/unitree-g1.md), including
  [installing the Unitree SDK](../hardware/unitree-g1.md#installing-the-unitree-sdk)
- [Microduck over the robotd link](../hardware/microduck.md)
- [Reachy Mini daemon link](../hardware/reachy-mini.md)

## Mounting a camera on a humanoid

`add_camera(parent_body=...)` mounts a camera ON a body so it rides with the
robot, and `position`/`target` are then in that body's LOCAL frame. The general
recipe in [World building](../simulation/world-building.md) reads the mount from
`list_bodies(robot_name=...)["gripper_body"]`, which is the right mount for an
arm. A humanoid here reports `gripper_body: None`: that hint set (`gripper`,
`hand`, `jaw`, `ee`, `tool`) is arm-shaped, and matching it on word boundaries is
what keeps a leg out of the answer - a `knee` link is not an end-effector because
`ee` occurs in its name. Pick the mount from the full `bodies` list instead.

For a head camera there is no head link to pick. The Unitree G1 asset's body tree
ends at the wrists - `pelvis` to hips/knees/ankles, and `waist_yaw_link` to
`waist_roll_link` to `torso_link` to shoulders/elbows/wrists - with no
`head_link`, `neck_link` or `eye_link`; its only sites are two IMUs and the two
feet. Mount on the torso with a local offset instead:

```python
sim = Robot("unitree_g1")
sim.add_camera(name="head", parent_body="unitree_g1/torso_link",
               position=[0.08, 0.0, 0.35], target=[1.0, 0.0, 0.2])
```

That puts the camera 0.35 m above the torso frame, roughly head height, and it
rides with the torso through waist yaw and roll. An arm camera mounts the same
way on a wrist link (`unitree_g1/left_wrist_yaw_link`). Bodies are namespaced by
the name passed to `Robot(...)`, so `Robot("g1")` would report `g1/torso_link`.

## See also

- [Mobile](mobile.md) - quadrupeds and wheeled bases.
- [Bimanual](bimanual.md) - two-arm rigs without the legs.
- [GR00T](../policies/groot.md) - many GR00T data_configs target humanoids.
