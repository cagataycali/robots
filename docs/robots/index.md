# Robot Catalog

38 robots across 7 categories. Every one works with the same `Robot("name")` interface.

---

## Categories

| Category | Count | Examples |
|---|---|---|
| [Arms](arms.md) | 16 | SO-100, Franka Panda, UR5e, xArm 7 |
| [Bimanual](bimanual.md) | 3 | ALOHA, Trossen WidowX AI, Bi-OpenArm |
| [Hands](hands.md) | 3 | Shadow Hand, LEAP Hand, Robotiq 2F-85 |
| [Humanoids](humanoids.md) | 8 | Unitree G1, Fourier N1, Apollo |
| [Mobile](mobile.md) | 7 | Spot, Unitree Go2, Stretch 3, Google Robot |
| Expressive | 1 | Reachy Mini |

---

## Quick Reference

```python
from strands_robots import Robot

# Arms
robot = Robot("so100")          # SO-100 tabletop arm
robot = Robot("franka_panda")   # Franka Emika Panda
robot = Robot("ur5e")           # Universal Robots UR5e

# Humanoids
robot = Robot("unitree_g1")     # Unitree G1
robot = Robot("fourier_n1")     # Fourier N1

# Bimanual
robot = Robot("aloha")          # ALOHA dual arm

# Mobile
robot = Robot("spot")           # Boston Dynamics Spot
robot = Robot("unitree_go2")    # Unitree Go2 quadruped
robot = Robot("google_robot")   # Google Robot (mobile manipulator)

# Hands
robot = Robot("shadow_hand")    # Shadow Dexterous Hand
```

Every robot auto-resolves from the registry. No configuration needed.
