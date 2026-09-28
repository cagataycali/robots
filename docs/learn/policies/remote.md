---
description: remote streams observations over a WebSocket to a PolicyServer on a GPU host and returns the action chunks it computes, so a CPU robot host runs any policy at control rate.
---

# remote

By the end of this page you can serve any policy from a GPU host with `PolicyServer` and drive a robot from a CPU host with `RemotePolicy`, using the same `run_policy` call you use for a local provider.

```bash
pip install 'strands-robots[inference]'    # websockets only; numpy-agnostic, composes with lerobot
```

## What it is

`RemotePolicy` is a `Policy` whose `get_actions` forwards each observation to a `PolicyServer` over a WebSocket (WS-JSON) and returns the action chunk the server computed. The server wraps any other provider, so the pi0 or SmolVLA checkpoint stays on the machine with the GPU and the robot host installs only `websockets`. `create_policy` resolves any `ws://` or `wss://` string to this provider, and the client mirrors the served policy's `requires_images`, `execution_horizon`, `actions_per_step` and `supports_rtc`, so the runtime sizes chunks and skips camera rendering as it would for the policy itself. The connection opens on first use; building the client does not need the server up.

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

`endpoint` supersedes `host` and `port`; when it is absent the client dials `ws://host:port`. `host` is checked for delimiters and `port` must be an `int` in `[1, 65535]` before the URI exists, so a bad value is refused while you still hold it instead of surfacing later as an unreachable server. `connect_timeout` and `request_timeout` are seconds and must name a positive budget; `0`, a negative or `True` is a `ValueError` at construction.

## The server

```python title="sketch"
from strands_robots.inference import PolicyServer

# build the served policy by provider name ...
PolicyServer(policy_provider="lerobot/act_so101", host="0.0.0.0").serve()

# ... or hand it an object you already loaded
PolicyServer(policy=my_policy, port=8765).serve()
```

`PolicyServer` takes exactly one of `policy` or `policy_provider` (`policy_config` goes to `create_policy` for the second form). It binds `127.0.0.1` by default; set `host="0.0.0.0"` to accept other machines. `serve()` blocks; `start()` returns after binding and `stop()` closes the accepted connections, which is what a test or a notebook wants. `port=0` asks the OS for a free port and writes it back to `server.port`.

Transport auth and TLS are out of scope at this commit: put the link inside a tailscale or wireguard tunnel for anything beyond one LAN. The server serves one client at a time. The wrapped policy holds per-episode state (RTC chunk seams, diffusion RNG), so an internal lock serialises inference across connections and a second client waits.

## Real-Time Chunking end to end

The runner counts `rtc_observed_delay_steps` on the robot host, the client forwards it on every request, and the server applies it to the wrapped policy before it blends chunk seams. A policy that supports RTC behaves the same behind the WebSocket as it does in process. Round trip is the cost: the result's `avg_inference_ms` is network plus inference, so size `control_frequency` against it.

## Hardware

```python title="sketch"
from strands_robots import Robot, create_policy

arm = Robot("so101", mode="real", port="/dev/ttyACM0")
policy = create_policy("ws://gpu-box:8765")
arm.run_policy(policy, instruction="pick up the cube", duration=30.0)
```

On hardware, `run_policy` takes a policy object built with the same string, and the approval gate from [Agents](../agents.md) sits in front of it when an agent makes the call. When the server's policy `requires_images`, the robot host must attach the cameras the served policy was trained on; see [lerobot_local](lerobot-local.md) for camera keys.
