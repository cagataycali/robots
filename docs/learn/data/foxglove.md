# Foxglove

At the end of this page a simulated or real robot is a live data source in Foxglove, with its meshes moving in the 3D panel, every joint plotted and every camera streaming, the same session lands in an MCAP file you can scrub later, and you know what an operator can and cannot do from the Foxglove side.

Install the `[foxglove]` extra and add one keyword:

```python title="sketch"
from strands_robots import Robot

arm = Robot("so101", foxglove=True, foxglove_mcap="/tmp/so101.mcap")
print(arm.foxglove_url)      # ws://127.0.0.1:8765
arm.step(500)
```

Open Foxglove, choose Foxglove WebSocket and paste the URL, or use `arm.foxglove_link`, which opens the desktop app there. The same keyword works on a real arm, `Robot("so101", mode="real", driver="lerobot", port=..., foxglove=True)`, and `STRANDS_ROBOTS_FOXGLOVE=1` turns it on for every `Robot()` in a shell without touching the code. When port 8765 is busy the next free one is taken and reported; `foxglove="0.0.0.0:8765"` serves the LAN and `":0"` picks an ephemeral port.

## What you see

| topic | schema | what |
|---|---|---|
| `/tf` | FrameTransforms | every MuJoCo body pose, 50 Hz |
| `/<robot>/scene`, `/scene` | SceneUpdate | the robot's meshes and the task objects, drawn once and moved by `/tf` |
| `/<robot>/joint_states` | JointStates | positions and velocities |
| `/<robot>/camera/<name>` | CompressedImage | JPEG frames, 10 Hz |
| `/strands/log`, `/strands/events` | Log, JSON | what the bridge and its gate said |

The 3D view needs no URDF: a registry robot is a compiled MuJoCo model, and its visual geoms go to the panel as triangles, so an arm an agent has just spawned is drawn. A real arm gets joints, cameras and logs; its meshes stay with the digital twin. The publish happens on the same path as `ros2_bridge=True`, so both can be on at once.

## Recording and export

`foxglove_mcap=` writes the same channels to a new file (an existing path is refused). It is a debugging sidecar, not a dataset: with two 640x480 cameras at 30 Hz the file grows by about 75 MiB a minute, roughly 3.7 times what LeRobot v3 needs for the same frames, and a static scene is written once however many clients connect. A recorded episode goes the other way with one call:

```python
from strands_robots.foxglove import export_episode, mcap_info

export_episode("~/.cache/huggingface/lerobot/you/so101_reach", 0, "/tmp/ep0.mcap")
print(mcap_info("/tmp/ep0.mcap")["channels"])
```

The file carries `/observation/state` and `/action/state` in the `lerobot.Scalars` shape lerobot's own Foxglove layouts plot, one JPEG channel per camera, and `/lerobot/episode` naming the dataset, task and fps, all stamped from the dataset's timestamp column so the timeline scrubs.

## The gate

By default it advertises no capability: a client can watch; nothing it sends reaches the robot. `foxglove_services=True` adds one service, `strands/set_joint_positions`, taking `{"robot": "so101", "positions": {"1": 0.2}}`, and refuses a `robot` it does not drive. A call from a panel carries no operator approval, so it is refused with every other command surface's sentence until `STRANDS_FOXGLOVE_COMMAND_ALLOW=strands/set_joint_positions` (or `*`) is set in the robot's environment. Every decision is written to `/strands/events`.

The `status` action of the simulation tool and of the hardware tool reports the URL, so an agent can hand an operator the link without a new action. Foxglove 3.x speaks the `foxglove.sdk.v1` subprotocol the server offers; an older build fails to connect rather than connecting badly.
