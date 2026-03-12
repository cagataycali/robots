# Domain Randomization

Vary physics parameters during training so your policy works in the real world, not just in simulation.

---

## Why Randomize?

Simulation is never perfect. If you train only in one exact environment, the policy might fail on real hardware because:

- Friction is different
- Lighting changes
- Objects have different weights
- Camera angles shift

**Domain randomization** trains on *many* variations so the policy learns to handle anything.

---

## Physics Randomization

```python
from strands_robots import Simulation

sim = Simulation("so100", randomize={
    "friction": [0.5, 1.5],      # Range of friction coefficients
    "mass": [0.8, 1.2],          # ±20% object mass
    "damping": [0.9, 1.1],       # Joint damping variation
})
```

Each `reset()` samples new parameters from the specified ranges.

---

## Visual Randomization

```python
sim = Simulation("so100", randomize={
    "lighting": True,             # Random light direction and intensity
    "camera_position": 0.02,      # ±2cm camera position noise
    "texture": True,              # Random surface textures
})
```

---

## Cosmos Visual Transfer

For photorealistic sim→real transfer, use NVIDIA Cosmos:

```bash
pip install "strands-robots[cosmos-transfer]"
```

```python
from strands_robots import cosmos_transfer

# Transform simulation images to look like real-world images
real_looking = cosmos_transfer(sim_image, style="real_kitchen")
```
