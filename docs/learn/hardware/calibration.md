---
description: One calibration file gives every joint reading the same meaning here, in lerobot and in recorded datasets.
---

# Calibration

At the end of this page every joint reading and every commanded degree on a serial arm means the same thing in this package, in lerobot and in your recorded datasets, because all three read one calibration file, and you know what the package does when that file is missing.

Calibration is lerobot's procedure, run once per arm from the shell with the arm plugged in:

```bash
pip install 'strands-robots[lerobot]'
lerobot-find-port                                   # which /dev/tty* is the arm
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0     # first time only: assign servo ids
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_arm
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
```

You move each joint to both stops; lerobot records the encoder counts and writes one JSON per device under `HF_LEROBOT_CALIBRATION` (default `~/.cache/huggingface/lerobot/calibration/`), laid out as `robots/<type>/<id>.json` and `teleoperators/<type>/<id>.json`.

## Who reads the file

| path | how it finds the calibration |
|---|---|
| `Robot("so101", mode="real", port=..., id="my_arm")` (lerobot driver) | by `id`; `calibration_dir=` overrides the directory. `id` defaults to the tool name |
| `Robot("so101", mode="real", driver="strands", calibration=...)` (native) | `calibration=` is the path or the loaded records; `lerobot_calibration_path("so101_follower", "my_arm")` builds the path from lerobot's own constants |
| `pose_tool(..., calibration="/path.json")` | reads the file, records nothing |
| `Teleoperator("so101_leader", port=..., id="my_leader")` | by `id`, through lerobot |

```python title="sketch"
from strands_robots.drivers.feetech.bus import lerobot_calibration_path, load_calibration

path = lerobot_calibration_path("so101_follower", "my_arm")      # honours HF_LEROBOT_CALIBRATION
records = load_calibration(path)                                  # {motor: MotorCalibration}
arm = Robot("so101", mode="real", driver="strands", port="/dev/ttyACM0", calibration=records)
```

`load_calibration` refuses a file with a missing or non-integer field instead of filling it from defaults: a completed record would report degrees against a travel nobody measured. Both names passed to `lerobot_calibration_path` must be bare path segments; `..` or a separator is refused, since both arrive from tool calls.

## Without a file

The two drivers behave differently, and both say so:

- **lerobot driver.** lerobot itself prompts to calibrate on `connect()`. In a `lerobot_teleoperate` session `auto_accept_calibration` answers that prompt. The read-only `get_state` action does not connect through lerobot; it reads raw ticks over the bus and reports degrees estimated as `(ticks - 2048) * 360 / 4096`, labelled an estimate, so an agent can report the arm uncalibrated instead of failing to read it. A calibration flag that could not be read is reported as unread, not as uncalibrated.
- **native driver.** With no `calibration=` the bus uses `full_travel_calibration`: count 0 is one end of the servo's rotation and 4095 the other. Readings and targets are then off by however far your arm's mechanical stops sit inside that rotation. `get_status()` reports `calibration_source: None` so you can see which is in force.

## Why it matters for data

A dataset recorded with one calibration and replayed on an arm with another moves to different physical positions for the same numbers. Keep the `id` stable per physical arm, record it in the dataset's metadata ([record](../data/record.md)), and recalibrate only when you change a servo, which changes the counts at the stops.

## Other robots

Network arms (UR, Franka), DDS robots (Unitree, Booster) and daemon robots (Reachy Mini, Microduck) carry their calibration in their own controllers; the package reads joint values in the controller's units, nothing to calibrate. The Robotiq gripper runs its own open-close calibration stroke during activation, which is why `connect_eagerly()` waits for `gSTA == ACTIVE` before reporting connected.
