---
description: Reachy Mini bring-up over the daemon's /ws/sdk endpoint - what connect_eagerly reports, the whole action vocabulary, and what an accepted command does not prove.
---

# Reachy Mini daemon link

The Reachy Mini has no lerobot robot type, so its
[native driver](native-drivers.md) is the only way to reach it.

## Connecting

Install `strands-robots` and `websockets>=17.0`. The driver uses the daemon's
`/ws/sdk` endpoint on **both Lite and Wireless** hardware (verified against
Wireless daemon **1.10.0**) and needs no LeRobot, local Reachy SDK or Device
Connect bridge. A caller supplying `transport=` still selects its explicit Zenoh
bridge, which is the path an older daemon without `/ws/sdk` needs; that path is
not covered by the 1.10.0 hardware proof.

```python
from strands import Agent
from strands_robots import Robot

agent = Agent(tools=[Robot("reachy_mini", mode="real")])
agent("look at me, then say hello and turn toward whoever talks")
```

`mode="real"` with no port discovers the daemon (`$REACHY_HOST`, then
`localhost`, then `reachy-mini.local`; a desktop daemon reporting its own start-up
error is skipped), `mode="auto"` asks the same probe before falling back to sim,
and the first verb that needs the daemon connects. `port="reachy-a.local:8000"`
skips discovery; `mini.cleanup()` closes this client's link, not the daemon or its
motors.

Streams arrive asynchronously, so a cache can still be `None` right after
connecting, and `status` reports this client's bookkeeping rather than a fresh
health probe - caches are last-received samples, not proof the robot is reachable
now. The seven daemon head-motor values are **body yaw then six Stewart legs**,
separated by `joints.body_yaw_deg` and `joints.head_leg_deg`;
`joints.antennas_deg` is **[right, left]**, as on the wire, and a six-leg legacy
bridge has no measured body yaw (`None`). `pose` is the head IMU orientation,
**not** the daemon's kinematic 4x4 head pose, and battery is `None` when the
payload carries no percentage.

## The action vocabulary

The native tool declares one `action` per daemon path:

| Action | What it does |
|---|---|
| `sensors` / `get_state` | cached IMU, head pose, battery, joints (six head legs, `body_yaw`, antennas; degrees) |
| `status` | daemon reachability, hardware variant, connection error |
| `stop` | stops every running move by uuid, and the `turn_to_sound` loop |
| `look` | a smooth head pose over `duration`: `pitch`/`roll`/`yaw` deg, `x`/`y`/`z` mm |
| `antennas` | the ears alone (`antenna_right`/`antenna_left`, deg) |
| `body_turn` | rotates the body (`body_yaw`, +/-160 deg) |
| `home` | neutral pose, antennas level, body centred |
| `wake` / `sleep` | the daemon's wake-up / go-to-sleep choreography |
| `express` | a recorded emotion or dance; plain words (`happy`, `curious`, `no`) resolve to library moves |
| `list_moves` | the library's move names (`library=emotions` or `dances`) |
| `motors` | torque mode (`enabled`, `disabled`, `gravity_compensation`), or omit `mode` to read it |
| `say` | speaks text through a TTS sidecar (`tts_url=` or `REACHY_TTS_URL`, else refused by name) |
| `play_sound` | plays a WAV the daemon can read |
| `volume` / `set_volume` | reads / sets the speaker level; the daemon also plays a test sound, so `allow_test_sound=true` |
| `track_face` | daemon face tracking on/off; at weight 1 it outranks `look` and `express` |
| `tracking_status` | whether a face is detected, and where |
| `camera` | saves one fresh camera JPEG locally |
| `record_audio` | records a bounded microphone WAV locally |
| `look_at` | turns the head toward a camera pixel (`u`, `v`, `frame_width`, `frame_height`) |
| `turn_to_sound` | faces whoever is talking, from the microphone array's direction of arrival |
| `turn_to_sound_status` | bearing, speech flag, turns, and why the last frame did not turn |

`say` and `play_sound` keep the head still unless `wobble=true` asks for the
daemon's audio-reactive wobbling, and the playback endpoint can answer `ok` with no
media server behind it - acceptance is not audible output.

Every write returns the daemon's acceptance and says `motion_verified=false`: it
does not prove a servo moved, that torque is enabled, or that another controller
will not overwrite it. The daemon accepts a move in every torque mode - with
torque off it answers with a move uuid and the head stays put - so `motors` with
no `mode` is how an accepted move that changed no pose gets an answer:

```python
agent("are your motors on?")   # motors -> {"motors": "disabled", "holds_a_pose": false}
```

In Python, `mini.send_action(...)` takes degrees and millimetres, prompts for no
approval, and sends whole groups: **both** antennas (an omitted side becomes zero)
or a whole head pose, never a delta. Obtain approval and exclusive motion
ownership first, keep HITL gates in any agent-facing orchestration, and restore
the starting state afterwards.

## Camera and microphone capture

Both go through the daemon's GStreamer WebRTC service rather than a dashboard or
the Reachy SDK, and only they need it installed: **PyGObject** and **GStreamer**
with the `rswebrtc`, JPEG and video-conversion plugins, under an interpreter that
imports `gi.repository.Gst` and `gi.repository.GstApp` (on macOS an isolated
Python may need `DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix)/lib"`). Signaling
defaults to port **8443** (`media_port=` overrides it) and selects one producer
named `reachymini`; caller and robot must share a trusted network, and an
authenticated or TLS daemon refuses capture rather than bypassing those settings.

`mini.capture_frame(save_path="")` and `mini.record_audio(duration=0.5,
save_path="")` are the direct equivalents. Empty paths create private temporary
files, explicit paths must be new and never overwrite a file or symlink, and a
result carries the path, dimensions and source, **not image bytes**. A capture
whose receiver teardown cannot be confirmed refuses instead of returning data -
that receiver may still be active. A recording is **0.1-5 s** of mono,
16 kHz, signed 16-bit PCM, video discarded, one second of start-up audio dropped
inside the same budget, its duration measured from the samples rather than the
clock; reported discontinuities, gaps, corruption, short reads or cleanup failures
refuse without writing a WAV, and `transport_loss_verified` is always false,
because Opus concealment is not observable here.

## Pixel look-at

`mini.look_at(u, v, frame_width, frame_height, duration=1.0)` turns the head
toward a camera pixel. The pixel is resolved privately - daemon calibration, the
stream's crop factor, lens undistortion and a fresh head pose, GETs only - into a
head target that goes through the shared motion envelope as one smooth `goto`. A
pixel asking for more pitch than the platform has is refused, not clamped, and
only known full-sensor camera models (`wireless`, `lite`, `older_rpi`), advertised
unambiguous resolutions and **unmodified camera frames** are accepted. The result
reports the resolved geometry and the bounded target, and says
`safety_validated=false` and `frame_pose_synchronized=false` - the head pose was
sampled separately from the frame. There is no read-only planning verb.

## Acknowledged antenna targets

For supervised diagnostics, `mini.send_action({"antenna_right": right_deg,
"antenna_left": left_deg}, require_ack=True)` uses the native REST target handler
instead of fire-and-forget WebSocket commands. It requires both antenna values, a
connected driver, finite values and complete finite joint telemetry inside 0.5
monotonic seconds (wall-clock corrections do not change expiry); other axes are
refused, and default `send_action` is unchanged. Only an
explicit daemon `status=ok` is acknowledged - busy, unknown and failed responses
refuse without retrying another command path, and a timeout leaves delivery
uncertain. The mode adds no motion permission and no excursion bound: the
daemon's HTTP job list alone cannot establish exclusive control.

## What a failed bring-up reports

A native driver reports what it cannot reach rather than raising. If the Mini's
transport - a standard-library-only module of the core distribution, so no extra
decides whether it loads - cannot be imported, the driver still builds, registers
and answers `get_status`, and every surface that would touch the daemon returns a
reason naming the module and the error:

```python
>>> Robot("reachy_mini", mode="real").connect_eagerly()
"cannot import strands_robots.device_connect.reachy_transport: No module named
'strands_robots.device_connect.reachy_transport'"
```

No `pip install` is prescribed, because no install supplies a module that ships in
the core distribution, and the same reason arrives as `connect_error` in
`get_status`. A link that never finishes its handshake reports the same way,
naming the budget that expired rather than the timeout's empty message, and the
driver cancels the handshake and stops the link before returning:

```python
>>> Robot("reachy_mini", mode="real").connect_eagerly()
"link to reachy-a.local:8000 did not finish its handshake within 10s"
```

Either way the background asyncio loop is closed rather than merely stopped -
through `cleanup()` and on both give-up paths - so none leaks per connect cycle; a
thread outlasting the 5 s wait keeps its loop, and that is logged rather than
reported as a finished teardown.

The opt-in read-only hardware check never sends motion or STOP:

```bash
REACHY_TEST_READONLY=1 REACHY_TEST_HOST=reachy-a.local:8000 \
  python -m pytest tests_integ/drivers/test_reachy_native_hardware.py -q
```

The daemon link defaults to plaintext on a trusted LAN; the shared transport
honours `REACHY_DAEMON_TOKEN`, `REACHY_DAEMON_TLS` and certificate verification
when the daemon or its TLS proxy supports them. Never expose an unauthenticated
actuator endpoint to the Internet.

## See also

- [Native drivers](native-drivers.md) - the contract this driver satisfies.
- [Humanoids](../robots/humanoids.md) - the catalog entry, and the other
  native-driver bring-ups in this family.
- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
