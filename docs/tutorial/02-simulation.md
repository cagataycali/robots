---
description: The Simulation AgentTool — 60+ actions for worlds, cameras, objects, and randomization.
---

# 2 — Simulation

```python
from strands_robots import Robot

sim = Robot("so100")

# Free camera: world-space position + look-at target
sim.add_camera(name="side", position=[0.5, -0.5, 0.3], target=[0.0, 0.0, 0.1], fov=60)

# Add a physics object
sim.add_object(name="cube", shape="box", size=[0.025, 0.025, 0.025],
               position=[0.3, 0.0, 0.025], color=[1, 0, 0, 1])

result = sim.render(camera_name="side")          # PNG image content block (agent-ready)
frame  = sim.get_observation("so100")["side"]    # numpy uint8 HxWx3 for your own loop

sim.randomize(randomize_colors=True, randomize_lighting=True,
              randomize_physics=True, randomize_positions=True)

sim.set_gravity([0.0, 0.0, -3.7])   # Mars gravity
sim.step(n_steps=500)               # NOT num_steps; max 100 000/call

sim.run_policy(robot_name="so100", policy_provider="mock",
               instruction="pick up the cube", duration=10.0)
```

## Action vocabulary

| Group | Actions |
|-------|---------|
| **World** | `create_world`, `load_scene`, `reset`, `get_state`, `destroy` |
| **Robots** | `add_robot`, `remove_robot`, `list_robots`, `get_robot_state` |
| **Objects** | `add_object`, `remove_object`, `move_object`, `list_objects` |
| **Cameras** | `add_camera`, `remove_camera` |
| **Policies** | `run_policy`, `start_policy`, `stop_policy`, `eval_policy`, `replay_episode`, `run_multi_policy` |
| **Rendering** | `render`, `render_depth`, `open_viewer`, `close_viewer` |
| **Physics** | `step`, `set_gravity`, `set_timestep`, `get_contacts` |
| **Recording** | `start_recording`, `stop_recording`, `get_recording_status`, `start_cameras_recording`, `stop_cameras_recording` |
| **Scene MJCF** | `replace_scene_mjcf`, `patch_scene_mjcf` |
| **Randomization** | `randomize` |
| **Assets** | `list_urdfs`, `register_urdf`, `get_features` |

Key notes:
- `add_camera` only creates free cameras. Robot-mounted cameras come from the URDF and are auto-discovered; list with `get_features(robot_name=...)`.
- `add_object` shapes: `"box"` `"sphere"` `"cylinder"` `"plane"` `"mesh"`. Plane requires `is_static=True`.
- Multi-robot scene: call `add_robot(robot_name="so100", position=[...])` on the existing sim, or `load_scene(scene_path=...)`.

## See also

- [Tutorial 3 — Policies](03-policies.md) — drop a real policy into `run_policy`.
- [Simulation overview](../simulation/overview.md) — every action with full parameters.
- [World building](../simulation/world-building.md) — non-trivial scene composition.
- [Domain randomization](../simulation/domain-randomization.md) — what `randomize` samples.
