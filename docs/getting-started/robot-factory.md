---
description: Robot(name, mode, backend, urdf_path, cameras, position, data_config, mesh, peer_id, **kwargs) — the full signature with every kwarg explained.
---

# Robot factory

The single entrypoint to the library. `Robot(...)` is a function (not a class) that
returns a concrete `Simulation` or `HardwareRobot` instance.

```python
from strands_robots import Robot

robot = Robot(
    name="so100",
    mode="sim",
    backend="mujoco",
    urdf_path=None,
    cameras=None,
    position=None,
    data_config=None,
    mesh=True,
    peer_id=None,
)
```

## The signature

| Param | Type | Default | What |
|-------|------|---------|------|
| `name` | str | required | Catalog name or alias. Resolved through `registry/robots.json`. |
| `mode` | str | `"sim"` | One of `"sim"`, `"real"`, `"auto"`. Case-insensitive. |
| `backend` | str | `"mujoco"` | Sim backend. Only `"mujoco"` today; `"isaac"` / `"newton"` are roadmap. Ignored when `mode="real"`. |
| `urdf_path` | str | `None` | Explicit MJCF/URDF path. Bypasses the registry asset block. Required for unknown names. |
| `cameras` | dict | `None` | Real-hardware camera config. Sim cameras: use `add_camera` after construction. |
| `position` | list | `None` | Robot position `[x, y, z]` in the sim world. |
| `data_config` | str | `None` | GR00T data_config name. Defaults to canonical robot name. |
| `mesh` | bool | `True` | Auto-join the Zenoh mesh. |
| `peer_id` | str | `None` | Stable mesh peer id. Auto-generated if omitted. |
| `**kwargs` | | | Forwarded to the backend constructor. |

## Modes

### `"sim"` (default — safe)

Returns a `strands_robots.simulation.Simulation`. No hardware touched. Safe to call in
CI, demos, notebooks.

```python
sim = Robot("so100")
print(type(sim).__name__)          # 'Simulation'
print(sim.tool_name_str)           # 'so100_sim'
```

### `"real"` (explicit hardware)

Returns a `strands_robots.hardware_robot.Robot`. Requires `[lerobot]` extra plus a
calibrated, USB-connected arm.

```python
robot = Robot("so100", mode="real",
              cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0"}})
```

If the controller isn't found or calibration is missing, `Robot()` raises before any
motion.

### `"auto"`

Probes USB for known servo controllers. If found → `"real"`. If not → `"sim"`. Useful
in scripts that should "just work" on a laptop and a real-hardware rig with the same
code:

```python
robot = Robot("so100", mode="auto")
```

The detection is shallow — see `_auto_detect_mode` in `strands_robots/robot.py`. For
production deployments, prefer explicit `mode="real"` or `mode="sim"`.

## Environment override

`STRANDS_ROBOT_MODE` overrides the `mode` kwarg when set:

```bash
export STRANDS_ROBOT_MODE=sim
python my_script.py     # all Robot() calls inside force mode='sim'
```

Useful for CI ("force everything sim regardless of what the script asks for").

## Name resolution

```python
from strands_robots.registry import resolve_name

resolve_name("SO-100")    # 'so100'
resolve_name("franka")    # 'panda' (alias)
resolve_name("g1")        # 'unitree_g1' (alias)
resolve_name("h1")        # 'unitree_h1' (alias)
```

Case-insensitive, hyphens and underscores interchangeable.

The full alias map lives in `registry/robots.json`. To add an alias, edit the JSON.

## Validation

`Robot()` refuses unknown names unless you also pass `urdf_path`:

```python
Robot("not_a_real_robot")
# ValueError: Unknown robot 'not_a_real_robot' and no urdf_path provided.

Robot("my_custom_arm", urdf_path="/path/to/arm.xml")
# OK — explicit URDF/MJCF path bypasses the registry
```

Empty / whitespace-only names always raise.

## Forwarding `**kwargs`

Anything not consumed by the factory itself is forwarded to the backend:

```python
# Sim — these go to Simulation.__init__
sim = Robot("so100",
            default_timestep=0.005,
            default_width=1280,
            default_height=720)

# Real — these go to HardwareRobot.__init__
hw = Robot("so100", mode="real",
           port="/dev/tty.usbserial-A50285BI",
           control_frequency=50.0)
```

Unknown kwargs raise a clear error from the backend constructor — they're not silently
dropped.

## Mesh wiring

Every `Robot()` joins the Zenoh mesh by default. The `.mesh` attribute on the returned
instance gives access to peer discovery and RPC:

```python
sim = Robot("so100")
print(sim.mesh.peer_id)     # 'so100_sim-a1b2c3d4'
print(sim.mesh.alive)       # True
print(sim.mesh.peers)       # other peers on the LAN
```

Disable per-robot:

```python
sim = Robot("so100", mesh=False)
print(sim.mesh)             # None
```

Or process-wide:

```bash
export STRANDS_MESH=false
```

If the mesh fails to start (zenoh missing, port bound, etc.) the factory still returns
a working sim/hardware robot with `.mesh = None` — mesh failure is non-fatal.

## See also

- [Tutorial 9 — Advanced](../tutorial/09-advanced.md) — call sequence + extension points.
- [Robot catalog](../robots/index.md) — the 68 catalog names.
- [Architecture](../architecture.md) — where the factory sits in the module map.
- [Tutorial 5 — Multi-robot](../tutorial/05-multi-robot.md) — mesh peer discovery and RPC.
