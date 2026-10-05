---
description: Cameras on a real robot for policies and recordings, one frame from an agent, and the same camera in simulation.
---

# Cameras

This page attaches cameras to a real robot for policies and recordings, reads one frame without moving, discovers what is plugged in, and mirrors the camera in simulation so one policy runs on both.

This runs without hardware:

```python
from strands_robots import Robot

sim = Robot("so101")
sim.add_camera("front", position=[0.6, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=320, height=240)
frame = sim.render(camera_name="front")
print(frame["content"][0]["text"])          # 320x240 from 'front' at t=0.000s
# frame["content"][1]["image"] is a PNG the agent can see
```

## Real cameras on a robot

Cameras are a constructor argument, one entry per name, and the `type` is a lerobot camera class:

```python title="sketch"
arm = Robot(
    "so101", mode="real", port="/dev/ttyACM0",
    cameras={
        "front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 30},
    },
)
```

| `type` | class | notes |
|---|---|---|
| `opencv` | `OpenCVCameraConfig` | USB and built-in cameras; `index_or_path` is an index or a device path |
| `intelrealsense` | `RealSenseCameraConfig` | `serial_number_or_name`; needs the Intel SDK on top of lerobot. The spelling `realsense` is refused with a hint |

Every other key must be a declared field of the resolved config class; a typo is refused by name. The camera names become the `observation.images.<name>` columns a recording writes and a policy reads, so match them to the `data_config` you train with (`so100_dualcam` names `front` and `wrist`).

Native drivers address cameras through their own SDK (Reachy Mini, EarthRover `camera` verb, Microduck) and do not read `cameras=`; passing a non-empty dict to one is refused unless the class declares `reads_cameras = True`. One that does retries a mode that yields no frame without its `fps`, then with no size (some UVC cameras accept a rate they never deliver), and its status row names it `refused_mode`.

## Look without moving

The real robot tool exposes `list_cameras` and `render` next to `execute`. Neither writes a servo register: `render` opens the camera under the bus lock and returns one PNG, `list_cameras` reports each configured camera and whether it is open. An agent asked to "look at the table" needs no rollout.

## Discover and test

`lerobot_camera` is the agent-facing tool for the camera side of a rig, independent of any robot. It needs `[lerobot]`:

```python title="sketch"
from strands_robots import lerobot_camera

lerobot_camera(action="discover")                                       # OpenCV and RealSense devices
lerobot_camera(action="test", camera_id=0, fps=30)                      # frame rate and latency
lerobot_camera(action="capture", camera_id=0, save_path="./captures")  # one image, returned inline
lerobot_camera(action="capture_batch", camera_ids=[0, "/dev/video2"])
```

Actions: `discover`, `list`, `capture`, `capture_batch`, `record`, `preview`, `test`, `configure`. `filename`, `format` and `save_path` are resolved inside `save_path`; a value that escapes it is refused. `color_mode` is `RGB` or `BGR`, `rotation` is one of `NO_ROTATION`, `ROTATE_90`, `ROTATE_180`, `ROTATE_270`; any other spelling is refused rather than read as a default.

## Simulation cameras

In `mode="sim"` cameras are not a constructor argument; add them afterwards with `add_camera(name, position=, target=, fov=60.0, width=640, height=480)` or through the robot tool's `add_camera` action. `render(camera_name=)` returns the PNG, `render_depth` the depth map, `render_all` every camera, `start_cameras_recording` writes each to an MP4 ([record](../data/record.md)). The `default` camera always exists.

A sim camera named `front` produces `observation.images.front`, the same column a real `front` camera does.

## On the mesh

A robot with cameras publishes frames on its camera topic when the mesh is on; the IoT leg can offload them to S3 (see [bridges](../mesh/bridges.md)). Camera topics are the heaviest thing on the mesh, so the publish rate is bounded separately from joint state.
