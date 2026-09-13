# Remote (WebSocket inference)

`create_policy("remote", endpoint="ws://gpu-box:8765")` returns a
`RemotePolicy` - a drop-in [`Policy`](overview.md) that forwards observations to
a remote `PolicyServer` and returns the action chunk it computes. Use it when
the robot host cannot run a large policy locally and a GPU box can.

```python
from strands_robots import create_policy

policy = create_policy("remote", endpoint="ws://gpu-box:8765")
# smart string is equivalent:
policy = create_policy("ws://gpu-box:8765")
```

| Config key        | Default       | Meaning                                          |
|-------------------|---------------|--------------------------------------------------|
| `endpoint`        | -             | Full server URL (`ws://host:port`); wins over host/port |
| `host`            | `127.0.0.1`   | Server host (when `endpoint` is omitted)         |
| `port`            | `8765`        | Server port (when `endpoint` is omitted)         |
| `connect_timeout` | `10.0`        | Seconds to wait for the WebSocket handshake      |
| `request_timeout` | `60.0`        | Seconds to wait for each inference reply         |

Both budgets are deadlines on a read, so a server that accepted the connection
and then went quiet - a checkpoint still loading onto the GPU, a wedged forward
pass - returns control to the caller instead of holding it. That case is
reported apart from an absent server, because the remedies differ: an absent
server is told to start one, while a listening one names the read that expired,
the budget that expired and the parameter carrying it, so the choice between
reading the server's log and raising the budget is the operator's to make.

The client mirrors the server policy's `requires_images`, `execution_horizon`,
`actions_per_step`, `supports_rtc` and `required_bodies`, and forwards the
Real-Time Chunking observed-delay count on every request, so a remote rollout
behaves like a local one. Mirroring `required_bodies` is what lets the robot
host supply a body pose the remote policy needs - and refuse a body name its
scene does not contain - since both of those happen locally. Install with `pip install 'strands-robots[inference]'`.

See **[Remote Policy Inference](../inference/remote.md)** for the full
two-machine setup, the server CLI, and the wire protocol.
