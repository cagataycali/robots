---
description: Dexterous end-effectors - Allegro, Shadow, LEAP, Robotiq, etc.
---

# Hands

Dexterous end-effectors - Allegro, Shadow, LEAP, Robotiq, etc.

```python
from strands_robots import Robot
sim = Robot("shadow_hand")      # Shadow Hand
sim = Robot("leap_hand")        # LEAP Hand
sim = Robot("robotiq_2f85")     # Robotiq 2F-85 gripper
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:hand}}

## Hardware control

Every entry above builds in simulation. One of them also drives real hardware:
the Robotiq 2F-85 has a native Modbus TCP driver, so `mode="real"` reaches the
gripper itself.

```python
from strands_robots import Robot

gripper = Robot("robotiq_2f85", mode="real", port="192.168.1.11")
gripper.connect_eagerly()          # opens the socket AND activates the gripper

gripper.send_action({"gripper": 1.0})        # 0.0 fully open, 1.0 fully closed
gripper.send_action({"aperture_mm": 40.0})   # or millimetres between the fingertips
gripper.read_status()                        # aperture, grasp detection, faults
```

`driver="strands"` is the default for this robot (its registry entry declares
it), because lerobot has no robot type for a gripper.

Two things worth knowing about the hardware:

- **Activation is not optional.** A 2F-85 powers up unactivated and *ignores*
  every position command until it has run a calibration stroke - it reports no
  error, it simply does not move. `connect_eagerly()` performs the activation
  sequence and waits for it to finish, so a successful connect means a gripper
  that will actually move.
- **A grasp is not the same as a stop.** `read_status()` reports `holding`,
  derived from the gripper's `gOBJ` field: the fingers stopping because they
  reached the commanded position is a different outcome from stopping because
  something is between them. Reading only "stopped" reports every empty close
  as a successful pick.

The 2F-140 and Hand-E answer the same register map with a different stroke; pass
`stroke_mm=` to drive one.

## See also

- [Arms](arms.md) - pair a hand with an arm via `add_robot`.
- [Custom policies](../policies/custom-policies.md) - high-DOF hand control needs careful action-space design.
- [Bimanual](bimanual.md) - two arms each with a hand.
