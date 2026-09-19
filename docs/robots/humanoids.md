---
description: Full-body humanoids and expressive desktop robots.
---

# Humanoids

Full-body humanoids and expressive desktop robots.

```python
from strands_robots import Robot
sim = Robot("unitree_g1")       # Unitree G1
sim = Robot("unitree_h1")       # Unitree H1
sim = Robot("apollo")           # Apptronik Apollo
sim = Robot("reachy_mini")      # Pollen Reachy Mini (expressive)
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:humanoid, expressive}}

## Real hardware: the Booster T1 native driver

The T1 is driven natively through its own SDK
(`booster_robotics_sdk_python`, a pybind11 wrapper over the robot's DDS
transport) — lerobot has no robot type for it, so `driver="strands"` is the only
way to reach it and the registry declares it as the default:

```python
from strands_robots import Robot

t1 = Robot("booster_t1", mode="real", port="192.168.10.102")  # driver="strands"
t1.connect_eagerly()          # ChannelFactory + loco client + LowState/LowCmd channels

t1.move(vx=0.2)               # walk: a twist to the onboard controller
t1.rotate_head(pitch=0.1, yaw=-0.3)

t1.enable_upper_body(True)    # claim the arms (UpperBodyCustomControl)
t1.send_action({"left_shoulder_pitch": -0.4, "right_elbow_pitch": 0.7})
t1.stop_task()                # halt locomotion + hand the arms back
```

Control of the T1 is **split**, and the driver enforces the split rather than
documenting it. An onboard whole-body controller owns the legs, waist and head;
the eight upper-body joints can be handed to a host, and only then:

| Slots | Joints | Who commands them |
|-------|--------|-------------------|
| 2–9 | `left`/`right` `shoulder_pitch`, `shoulder_roll`, `elbow_pitch`, `elbow_yaw` | the host, via `send_action`, after `enable_upper_body(True)` |
| 0–1 | `head_yaw`, `head_pitch` | the robot, via `rotate_head()` |
| 10–22 | waist, hips, knees, ankle cranks | the robot, via `move()` |

Every frame the driver publishes puts `kp=kd=0` on every slot outside 2–9, which
is what leaves the onboard controller in charge of balance; an uncommanded arm
joint holds its last *observed* position, so commanding one arm does not drop the
other. `send_action` refuses before the gate is open, before the first `LowState`
has arrived (the frame width and the hold positions both come from the robot's
own report), and for any joint outside the upper body — naming `move()` or
`rotate_head()` instead.

The driver also subscribes the T1's battery and fall-down topics. A fall state
other than `IS_READY` refuses a write — a held arm posture is noise while the
robot is on its way to, on, or getting off the floor, and can obstruct its
getting-up routine — and the gate reads *evidence of a fall* rather than the
absence of a reading, so a T1 whose fall topic is silent keeps writing. The
charge read reaches `get_status()` as the shared `battery_pct` field and gates
nothing: the SDK names it `soc` and documents no scale, and a floor compared
against an unverified scale refuses every frame or none while looking like a
working check.

`run_policy`/`start_task` refuse: this driver publishes one frame per call and
owns no control loop. A caller who wants a trajectory calls `send_action` on
their own timer (the vendor's reference client runs 100 Hz).

The SDK is a vendor wheel rather than a declared dependency of this project
(`pip install booster_robotics_sdk_python`, linux wheels only) — the same footing
as the G1's `unitree_sdk2py` ([recipe](#installing-the-unitree-sdk)). Without it the driver still imports, builds and
answers `get_status`; `connect_eagerly()` returns a reason naming the module and
the install line. Because the wheel is pinned to the robot's firmware
rather than resolved by this project, the *installed* build's vocabulary is an
input: `ROBOT_MODES` and the two `cmd_type` conventions are names this driver
spells so a caller sees them without the SDK, and a build that declares a
different set is refused by name — naming the enum, the member and the modes
that build does have — rather than raising out of the verb.

## Real hardware: the Unitree G1 native driver

The G1 has no lerobot robot type either, so `mode="real"` builds the native
CycloneDDS driver in `strands_robots.drivers.g1` (the registry declares
`hardware.driver = "strands"`):

```python
from strands_robots import Robot

g1 = Robot("g1", mode="real", port="192.168.123.161")   # network_interface="eth0" by default
g1.connect_eagerly()      # subscribes rt/lowstate, bms, lidar, mainboard; None when the bus is up
await g1.get_status()     # connection, FSM, battery
```

The driver-as-tool is deliberately small - `sensors`, `status`, `stop` - so an
agent can introspect the robot the day it is built. Motion goes through the
FSM-gated `g1_tools` bundle (`g1_send_action`, `g1_run_policy`, `g1_start_task`,
the `g1_safe_*` posture verbs) and, for the raw SDK, `use_unitree`; see the
[hardware tools](../hardware/tools.md) and [security](../security.md) pages.

### Installing the Unitree SDK

`unitree_sdk2py` is Unitree's vendor SDK and is **not** an extra of this
project. It cannot honestly be one: the PyPI `unitree-sdk2` 1.0.1 wheel ships no
`g1` or `comm` package (its `__init__` imports a `b2` it does not contain, so
`import unitree_sdk2py` fails) and pins `cyclonedds==0.10.2`, whose wheels stop
at Python 3.10 - under this project's `requires-python = ">=3.12"` that is a
source build that wants the CycloneDDS C library. Without the SDK the driver
still imports and builds; `connect_eagerly()` and every write verb return a
refusal that names this recipe.

A *partial* install fails elsewhere, and that is the shape the PyPI wheel
produces. With the bus bindings and the IDL types present but no `comm`
package, `connect_eagerly()` **succeeds** and the only thing that fails is the
motion-switcher open - reported as `motion_switcher_open_error` by the G1's
`get_status()`, and as the refusal from the Go2's `release_sport_mode()`. Both
name the same recipe and keep the SDK's own exception, so the module that is
actually missing is in the text.

The upstream checkout installed beside a `cyclonedds` wheel is what works, and
the binding comes from this project's `[ros2]` extra - the one place the range is
declared. On macOS arm64 and x86_64 Linux (Python 3.12):

```bash
pip install 'strands-robots[ros2]'                # the cyclonedds binding
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install --no-deps -e ./unitree_sdk2_python    # --no-deps skips the ==0.10.2 pin
python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('ok')"
```

On Linux aarch64 - the Jetson the robot ships with - `cyclonedds` publishes no
wheel at any version, so the C library comes first and the binding is built
against it:

```bash
git clone --branch 0.10.2 --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cdds
cmake -S /tmp/cdds -B /tmp/cdds/build -DCMAKE_BUILD_TYPE=Release && sudo cmake --build /tmp/cdds/build --target install
export CYCLONEDDS_HOME=/usr/local
pip install 'cyclonedds==0.10.2'
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install --no-deps -e ./unitree_sdk2_python
```

0.10.2 is the CycloneDDS release Unitree's own images and SDK pin, and the one
the `g1_tools` bundle was developed against. The 11.x wheel imports, builds the
IDL types and binds a `ChannelFactory`, but it has not been proven against a
live G1 bus; if the robot's topics stay silent under 11.x, build 0.10.2 as
above. Point `CYCLONEDDS_URI` at the robot's `cyclonedds.xml` when the default
multicast discovery does not find it.

## Real hardware: the Microduck robotd driver

The Microduck is driven natively through its on-robot
`robotd` daemon (Pollen's `duck-ipc-proto` JSON-RPC over a Unix socket, API
version 31 / microduck 0.14.1) — the same policy code that runs in sim drives
the physical robot. With no `port`, the driver finds the socket itself:

| Where you run | What to set | How it connects |
|---|---|---|
| On the duck | nothing | `/run/robotd.sock` |
| Another machine, key-authenticated ssh to the board | `MICRODUCK_HOST=[user@]<duck ip>` (user defaults to `radxa`, or `DUCK_BOARD_USER`) | the driver runs `ssh -N -L` and forwards robotd's, mediad's and tofd's sockets |
| You forwarded the socket yourself | `MICRODUCK_SOCKET=/path/to/local.sock` | that path |

```python
from strands_robots import Robot

duck = Robot("microduck", mode="real")            # discovers the socket (table above)
duck = Robot("microduck", mode="real", port="ssh://radxa@10.0.0.5")   # explicit forward

duck.connect_eagerly()                 # optional - the first verb connects on its own
duck.send_action({"vx": 0.15})         # walk forward (robot.move intent, one frame)
duck.send_action({"skill": "kick_left"})  # a named skill (robot.do)
duck.emergency_stop()                  # robot.stop
```

As an agent tool the driver exposes the whole operator vocabulary as one
`action`: `move` (a bounded twist kept alive past robotd's 0.5 s deadman and
ended with a zero twist), `head`, `look_at` (a point in the robot frame, solved
on the robot), `pose`, `mouth`, `do`/`skills` (the skills *this* robot lists),
`sit`/`stand`, `enable`/`disable`, `relax`/`init`/`reboot_motors` (each needs
`confirm=true`), `sounds`/`play_sound`, `theremin`, `mode`/`set_mode`
(walk/roller), `policies`/`load_policy`/`reload_policies`, `health`, `version`,
`model`, `odometry`, `monitor`, `camera` (one JPEG from mediad) and `tof` (one
depth-frame summary from tofd), beside the universal `sensors`/`status`/`stop`.
`Robot("microduck", mode="auto")` asks the driver's `probe_hardware()` first
(one Hello on the discovered socket) and is the real robot when a robotd
answers, MuJoCo when none does. A twist outside the pad's envelope (walk `|vx|,|vy| <= 0.3` m/s,
`|vyaw| <= 1.5` rad/s; roller `vx` in `[-0.5, 0.6]`, no strafe) is refused
before the wire, because robotd clamps nothing there; robotd's own
`accepted: false` comes back as the verb's refusal with its `reason`.

Every intent frame carries its whole group -- `robot.move` always carries
`vx`/`vy`/`vyaw`, `robot.pose` always `z`/`roll`/`pitch`/`active` -- so an
absent key is the resting value (`{"vx": 0.15}` walks straight ahead) and a key
this driver does not know is refused, even with a known key beside it. That
distinction matters because the two are indistinguishable on the wire:
`{"vx": 0.15, "yaw": 0.6}` -- `yaw` being the spelling the `get_status` pose
block uses for the heading -- would otherwise send `vyaw: 0`, walk straight past
the turn and report success. The same refusal catches a 14-joint
`MICRODUCK_JOINT_NAMES` action: four of its keys are head axes, so it would
arrive as a `robot.head` frame with the other ten joints dropped, which is the
per-joint stream `run_policy` refuses by name.

All three halt paths - `stop()`, `stop_task()` and `emergency_stop()` - send the
same `robot.stop`, and an accepted one is recorded in
`get_status()["motion_stopped"]`, the field an operator reads to decide whether
the robot is safe to approach. Only an accepted halt sets it: a stop robotd
declines leaves it false. `relax()` and `enable_torque(False)` de-energise rather
than halt a commanded motion, so they leave the flag alone.

`robotd` owns the walking/skill ONNX on-device, so `run_policy`/`start_task`
refuse and point back at the intent path; use `mode="sim"` for a host-driven
[`MicroduckPolicy` rollout](../policies/microduck.md#walking-in-mujoco).

## Mounting a camera on a humanoid

`add_camera(parent_body=...)` mounts a camera ON a body so it rides with the
robot, and `position`/`target` are then in that body's LOCAL frame. The general
recipe in [World building](../simulation/world-building.md) reads the mount from
`list_bodies(robot_name=...)["gripper_body"]`, which is the right mount for an
arm. A humanoid here reports `gripper_body: None`: that hint set (`gripper`,
`hand`, `jaw`, `ee`, `tool`) is arm-shaped, and matching it on word boundaries is
what keeps a leg out of the answer - a `knee` link is not an end-effector because
`ee` occurs in its name. Pick the mount from the full `bodies` list instead.

For a head camera there is no head link to pick. The Unitree G1 asset's body tree
ends at the wrists - `pelvis` to hips/knees/ankles, and `waist_yaw_link` to
`waist_roll_link` to `torso_link` to shoulders/elbows/wrists - with no
`head_link`, `neck_link` or `eye_link`; its only sites are two IMUs and the two
feet. Mount on the torso with a local offset instead:

```python
sim = Robot("unitree_g1")
sim.add_camera(name="head", parent_body="unitree_g1/torso_link",
               position=[0.08, 0.0, 0.35], target=[1.0, 0.0, 0.2])
```

That puts the camera 0.35 m above the torso frame, roughly head height, and it
rides with the torso through waist yaw and roll. An arm camera mounts the same
way on a wrist link (`unitree_g1/left_wrist_yaw_link`). Bodies are namespaced by
the name passed to `Robot(...)`, so `Robot("g1")` would report `g1/torso_link`.

## See also

- [Mobile](mobile.md) - quadrupeds and wheeled bases.
- [Bimanual](bimanual.md) - two-arm rigs without the legs.
- [GR00T](../policies/groot.md) - many GR00T data_configs target humanoids.


## Reachy Mini native camera

`Robot("reachy_mini", mode="real", driver="strands")` speaks the daemon
protocol directly. Its `camera` action captures a fresh JPEG through the daemon's
GStreamer WebRTC service, not through a dashboard or the Reachy SDK:

```python
from strands import Agent
from strands_robots import Robot

mini = Robot("reachy_mini", mode="real", port="reachy-a.local:8000", mesh=False)
try:
    error = mini.connect_eagerly()
    if error:
        raise RuntimeError(error)
    agent = Agent(tools=[mini], callback_handler=None)
    result = agent.tool.reachy_mini(action="camera")
finally:
    mini.cleanup()
```

Install **PyGObject** and **GStreamer**, including the `rswebrtc`, JPEG and video
conversion plugins, on the calling machine. These are optional system
requirements; ordinary telemetry needs neither. Use an interpreter that can
import both `gi.repository.Gst` and `gi.repository.GstApp`. On macOS, an isolated
Python may need `DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix)/lib"` at launch to
find Homebrew's native libraries; a pip-only install is not sufficient.
Native signaling defaults to
port **8443** (`media_port=` overrides it), selecting exactly one producer named
`reachymini`. The caller and robot must share a trusted network. Authenticated
or TLS daemon configurations currently refuse camera capture rather than
silently bypassing those settings.

`mini.capture_frame(save_path="")` is the direct equivalent. Empty paths create
private temporary files; explicit paths must be new, and never overwrite a file
or symlink. Results contain the path, decoded dimensions and source, **not image
bytes**. The Python capture polling wait is limited to ten seconds. Each call
requests teardown of its own receiver and waits up to three seconds to verify
successful completion with no pending state transition. It refuses to return
or save captured data if teardown cannot be confirmed; the receiver may still
be active after that error. These waits do not impose a hard deadline on native
GStreamer calls. Daemon media ownership is unchanged, negotiated microphone
audio is discarded, and no speaker audio or motor commands are sent.


### Bounded microphone capture

`mini.record_audio(duration=0.5, save_path="")` and the `record_audio` agent
action attempt a **0.1–5 second** local WAV recording. GStreamer converts incoming
audio to mono, 16 kHz, signed 16-bit PCM. Video is discarded and nothing is played.
One second of startup audio is discarded before recording, within the bounded
capture budget. Decoder-reported discontinuities, gaps, corruption, short reads
and cleanup failures refuse without saving a WAV. Duration comes from actual
PCM samples, not wall-clock time. GStreamer's presentation timestamps include
receive-clock corrections; those adjustments are returned under `quality`, not
misreported as lost samples or hidden by inserting/removing audio. Network/Opus
concealment is not fully observable at this boundary, so
`transport_loss_verified` is always false. Files have the same exclusive/private
policy as camera captures. Neither recording nor camera capture implies that
speaker playback or pixel look-at is implemented.


### Pixel look-at

`mini.look_at(u, v, frame_width, frame_height, duration=1.0)` - agent action
`look_at` with the same parameters - turns the head toward a camera pixel. It
is one verb with one meaning: the pixel is resolved privately (daemon
calibration, the stream's crop factor, lens undistortion and a fresh head pose,
GETs only) into a head target, that target's roll/pitch/yaw are put through the
shared motion envelope, and a single smooth `goto` is sent. A pixel that asks
for more pitch than the platform has is refused, not clamped. Only known
full-sensor camera models (`wireless`, `lite`, `older_rpi`) and advertised,
unambiguous resolutions are accepted, and pixels must come from an
**unmodified camera frame**, not a resized preview.

The result carries the resolved geometry under `plan` (`safety_validated=false`,
`frame_pose_synchronized=false` - the head pose was sampled separately from the
frame) and the bounded target under `target_deg`. `motion_verified` stays
`false`: the daemon accepted the move; nothing in the reply proves the head
arrived. There is no read-only planning verb and no `dry_run` switch.

### Acknowledged antenna targets

For supervised diagnostics, `mini.send_action({"antenna_right": right_deg,
"antenna_left": left_deg}, require_ack=True)` uses the native REST target handler
instead of fire-and-forget WebSocket commands. It requires both antenna values,
a connected driver, finite values and complete finite joint telemetry received
within 0.5 monotonic seconds (wall-clock corrections do not change expiry).
Other axes are refused in this opt-in mode; default `send_action` is unchanged.

Only an explicit daemon `status=ok` is acknowledged. Busy, unknown and failed
responses refuse, without retrying or falling back to another command path.
A timeout leaves delivery uncertain. Success says `motion_verified=false`:
an accepted target does not prove that a servo moved, that torque is enabled
for that motor, or that another controller will not overwrite it. This option
adds no motion permission or per-call excursion bound; supervision and encoder
readback remain necessary. The daemon's HTTP job list alone cannot establish
exclusive control.

### Sound, speech and volume

`play_sound(file)` posts the daemon's own playback endpoint; `say(text)` first
asks a TTS sidecar (`tts_url=` or `REACHY_TTS_URL`, refused by name when neither
is set) for a WAV the daemon can reach, then plays it. Both keep the head
still unless `wobble=true` asks for the daemon's audio-reactive head wobbling.
`volume` reads the level; `set_volume(level, allow_test_sound=true)`
changes it - the opt-in is required because the daemon plays a test sound on
every level change. The daemon's playback endpoint can return `ok` when no media
server exists, so acceptance is not evidence of audible output.

### The whole vocabulary, out of the box

```python
from strands import Agent
from strands_robots import Robot

agent = Agent(tools=[Robot("reachy_mini", mode="real")])
agent("look at me, then say hello and turn toward whoever talks")
```

`mode="real"` with no port discovers the daemon (`$REACHY_HOST`, then
`localhost`, then `reachy-mini.local`; a desktop daemon that reports its own
start-up error is skipped), `mode="auto"` asks the same probe before falling
back to sim, and the first verb that needs the daemon connects. The tool
declares `status`, `sensors`/`get_state`, `stop`, `camera`, `record_audio`,
`look`, `antennas`, `body_turn`, `home`, `wake`, `sleep`, `express`,
`list_moves`, `motors`, `say`, `play_sound`, `volume`, `set_volume`,
`track_face`, `tracking_status`, `look_at`, `turn_to_sound` and
`turn_to_sound_status`. `express` takes plain words (`happy`, `curious`, `no`)
as well as library names, from the emotions and dances libraries. `stop`
enumerates the daemon's running moves and stops each by uuid. Every write
returns the daemon's acceptance and says `motion_verified=false`.

The daemon accepts a move in every torque mode: commanded with torque off it
answers with a move uuid and the head does not move. `motors` with no `mode`
reports the mode the robot is actually in, so an accepted move that changed no
pose has an answer:

```python
agent("are your motors on?")   # motors -> {"motors": "disabled", "holds_a_pose": false}
agent("enable your motors and look up")
```

`motors(mode=...)` still sets it - `enabled`, `disabled` or
`gravity_compensation`.
