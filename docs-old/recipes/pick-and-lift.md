---
description: An SO-101 reaches, grips and lifts a cube in sim.
---

# Pick up a cube with an SO-101

You reach, close the gripper and lift a cube 15 cm off the table, in about a second on a CPU.

![The SO-101 reaching, gripping and lifting the cube](../assets/recipes/so101_pick_and_lift.png)

```python title="examples/18_so101_pick_and_lift.py"
--8<-- "examples/18_so101_pick_and_lift.py"
```

```console
$ MUJOCO_GL=egl python examples/18_so101_pick_and_lift.py
PICK OK - cube lifted 149.4 mm
```

Go deeper: [Objects](../reference/simulation/objects.md) · [Rollouts](../reference/simulation/rollouts.md)
