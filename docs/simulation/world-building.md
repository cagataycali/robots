# World Building

Create simulation environments with objects, obstacles, and terrain.

---

## Built-in Objects

```python
from strands_robots import Simulation

sim = Simulation("so100", objects=["red_cube", "blue_cylinder", "green_sphere"])
sim.reset()
```

Objects are placed in the scene and interact with the robot through physics.

---

## Custom Environments

Load custom MJCF scenes:

```python
sim = Simulation("so100", scene_path="/path/to/my_scene.xml")
```

---

## Object Placement

Control where objects appear:

```python
sim = Simulation("so100")
sim.add_object("red_cube", position=[0.3, 0.0, 0.05])
sim.add_object("blue_cube", position=[0.3, 0.1, 0.05])
sim.reset()
```
