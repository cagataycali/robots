---
description: Reachy Mini bring-up over the daemon's /ws/sdk endpoint - what connect_eagerly reports, which values the caches carry, and what the native agent tool exposes.
---

# Reachy Mini daemon link

The Reachy Mini has no lerobot robot type, so its
[native driver](native-drivers.md) is the only way to reach it.

## What a failed bring-up reports

A native driver reports what it cannot reach rather than raising. The Reachy Mini's
daemon transport is a standard-library-only module in the core distribution, so
nothing an extra installs decides whether it loads - but if it cannot be imported at
all, on a broken install or behind a shadowing module, the driver still builds,
registers and answers `get_status`, and every surface that would touch the daemon
returns a reason naming the module and the error instead:

```python
>>> Robot("reachy_mini", mode="real").connect_eagerly()
"cannot import strands_robots.device_connect.reachy_transport: No module named
'strands_robots.device_connect.reachy_transport'"
```

The reason stops at what it can establish. It prescribes no `pip install`, because no
install supplies a module that ships in the core distribution, and a remedy that
cannot help is worse than none - the same rule
[`require_optional`](https://github.com/strands-labs/robots/blob/main/strands_robots/utils.py) applies when it is told a module
arrives from a system package rather than an index.

The same reason arrives as `connect_error` in `get_status`, so a mesh peer for a Mini
whose transport will not load is still constructible and still reports why it is not
connected.

A bring-up that reaches the daemon but whose real-time link never finishes its
handshake is reported the same way, and the link is not left behind. The driver
cancels the handshake and asks the link to stop before returning, so nothing stays
subscribed to a Mini the caller has just been told it is not connected to:

```python
>>> Robot("reachy_mini", mode="real").connect_eagerly()
"link to reachy-a.local:8000 did not finish its handshake within 10s"
```

The reason names the budget that expired rather than the timeout's own message,
which is empty.

Either way the loop the bring-up opened is closed, not merely stopped. The link runs
on a background asyncio loop, and `loop.stop()` only asks it to return from
`run_forever` - the selector and self-pipe it opened are released by `loop.close()`.
So teardown waits for that thread (up to 5s) and then closes the loop, on the success
path through `cleanup()` and on both give-up paths, rather than leaving one open loop
per connect cycle for the garbage collector to complain about later. A thread that
outlasts the wait keeps its loop, because closing a running loop raises, and that
outcome is logged instead of reported as a teardown that finished.

## Connecting

Install `strands-robots` and `websockets>=17.0` in the client environment. The
native driver uses the daemon's `/ws/sdk` endpoint on **both Lite and Wireless**
hardware (verified against Wireless daemon **1.10.0**). It does not require
LeRobot, a local Reachy SDK, or a Device Connect bridge. Existing Wireless
callers supplying `transport=` still select their explicit Zenoh bridge.
Older daemons without `/ws/sdk` need that legacy path; they are not covered by
the 1.10.0 hardware proof.

```python
from strands import Agent
from strands_robots import Robot

mini = Robot("reachy_mini", mode="real", port="reachy-a.local:8000", mesh=False)
try:
    reason = mini.connect_eagerly()
    if reason is not None:
        raise RuntimeError(reason)
    agent = Agent(tools=[mini])
    # SDK tool dispatch without a model request. Reads only; no motion or STOP.
    print(agent.tool.reachy_mini(action="sensors"))
finally:
    mini.cleanup()  # closes the client link, not the daemon or its motors
```

Streams arrive asynchronously: immediately after connecting, a cache can still
be `None`. The seven daemon head-motor values are **body yaw followed by six
Stewart legs**; `joints.body_yaw_deg` and `joints.head_leg_deg` separate them.
`joints.antennas_deg` is **[right, left]**, matching the daemon wire protocol.
A legacy bridge supplying only six legs has no measured body yaw (`None`).
`pose` is the head IMU orientation, **not** the daemon's kinematic 4x4 head pose.
Battery is `None` when the status payload supplies no percentage. Status reports
this client's connection bookkeeping, not a fresh daemon health probe; caches
are last-received samples, not a guarantee that the robot is still reachable.

The registered native agent tool exposes only `sensors`, `status`, and `stop`.
The separate `reachy_*` helpers require a **live Python driver handle**, not a
handle an LLM can serialize. Native camera capture, audio playback, volume, and
pixel-directed look are not implemented; their helper tools return explicit
refusals. Recorded-move names must come from `mini.list_moves()`, not guessed
labels such as "happy".

For direct Python motion, `mini.send_action(...)` accepts degrees and millimetres.
It does not prompt for operator approval itself: obtain approval and exclusive
motion ownership first, and retain HITL gates in any agent-facing orchestration.
Do not test against an active voice, tracking, or autonomous controller. An
antenna command sends **both** antennas (an omitted side becomes zero); a head
command similarly sends a whole pose, not a delta. Capture the starting state,
use small bounded changes, and restore it after testing. `stop` requests a halt
of recorded moves only: it does not disable every independent controller.

The opt-in read-only hardware check never sends motion or STOP:

```bash
REACHY_TEST_READONLY=1 REACHY_TEST_HOST=reachy-a.local:8000 \
  python -m pytest tests_integ/drivers/test_reachy_native_hardware.py -q
```

The daemon link defaults to plaintext on a trusted LAN. The shared transport
honours `REACHY_DAEMON_TOKEN`, `REACHY_DAEMON_TLS`, and certificate verification;
configure these only when the daemon or its authenticated TLS proxy supports
them. Do not expose an unauthenticated actuator endpoint to the Internet.


## See also

- [Native drivers](native-drivers.md) - the contract this driver satisfies.
- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
