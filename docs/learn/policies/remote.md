---
description: remote streams observations over a WebSocket to a PolicyServer on a GPU host and returns the action chunks it computes, so a CPU robot host runs any policy at control rate.
---

# remote

By the end of this page you can serve any policy from a GPU host with `PolicyServer` and drive a robot from a CPU host with `RemotePolicy`, through the same `run_policy` call you use for a local provider.

```bash
pip install 'strands-robots[inference]'    # websockets only; composes with lerobot
```

## What it is

{{drawing:d09_remote_inference}}

`RemotePolicy` is a `Policy` whose `get_actions` forwards each observation to a `PolicyServer` over a WebSocket (WS-JSON) and returns the action chunk the server computed. The checkpoint stays on the machine with the GPU; the robot host installs `websockets`. `create_policy` resolves any `ws://` or `wss://` string to this provider, and the client mirrors the served policy's `requires_images`, `execution_horizon`, `actions_per_step` and `supports_rtc`, so the runtime sizes chunks and skips camera rendering as it would in process. The connection opens on first use.

```python
from strands_robots.inference import PolicyServer
from strands_robots.simulation import create_simulation

server = PolicyServer(policy_provider="mock", port=0).start()   # port=0 asks the OS for a free port
sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("so101")
result = sim.run_policy(
    robot_name="so101",
    policy_provider=f"ws://127.0.0.1:{server.port}",   # ws:// resolves to remote
    n_steps=20,
    control_frequency=50.0,
)
print(result["status"], result["content"][0]["text"])
server.stop()
sim.cleanup()
```

## Constructor keywords

{{providers:kwargs:remote}}

`endpoint` supersedes `host` and `port`; without it the client dials `ws://host:port`. `host` is checked for delimiters and `port` must be an `int` in `[1, 65535]` before the URI exists, so a bad value is refused while you still hold it. `connect_timeout` and `request_timeout` are seconds and must be positive; `0`, a negative or `True` is a `ValueError` at construction.

## The server

```python title="sketch"
from strands_robots.inference import PolicyServer

PolicyServer(policy_provider="lerobot/act_so101", host="0.0.0.0").serve()   # built by provider name
PolicyServer(policy=my_policy, port=8765).serve()                            # or an object you loaded
```

`PolicyServer` takes exactly one of `policy` or `policy_provider` (`policy_config` goes to `create_policy`). It binds `127.0.0.1`; set `host="0.0.0.0"` to accept other machines. `serve()` blocks; `start()` returns after binding and `stop()` closes accepted connections. `port=0` asks the OS for a free port and writes it back to `server.port`.

Transport auth and TLS are out of scope at this commit: run the link inside a tailscale or wireguard tunnel beyond one LAN. The server serves one client at a time; the wrapped policy holds per-episode state (RTC chunk seams, diffusion RNG), so a lock serialises inference and a second client waits.

## Real-Time Chunking end to end

The runner counts `rtc_observed_delay_steps` on the robot host, the client forwards it on every request, and the server applies it before the wrapped policy blends chunk seams. A policy that supports RTC behaves the same behind the WebSocket as in process. The result's `avg_inference_ms` is network plus inference; size `control_frequency` against it.

## Hardware

```python title="sketch"
from strands_robots import Robot, create_policy

arm = Robot("so101", mode="real", port="/dev/ttyACM0")
policy = create_policy("ws://gpu-box:8765")
arm.run_policy(policy, instruction="pick up the cube", duration=30.0)
```

On hardware `run_policy` takes a policy object built from the same string, and the [Agents](../agents.md) gate sits in front of it when an agent makes the call. When the served policy `requires_images`, attach the cameras it was trained on; [lerobot_local](lerobot-local.md) lists the camera keys.
