<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Robots
  </h1>

  <h2>
    Robot Control for Strands Agents
  </h2>

  <div align="center">
    <a href="https://pypi.org/project/strands-robots/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/strands-robots"/></a>
    <a href="https://github.com/strands-labs/robots"><img alt="GitHub stars" src="https://img.shields.io/github/stars/strands-labs/robots"/></a>
    <a href="https://github.com/strands-labs/robots/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-labs/robots"/></a>
    <a href="https://github.com/NVIDIA/Isaac-GR00T"><img alt="GR00T" src="https://img.shields.io/badge/NVIDIA-GR00T-76B900?logo=nvidia"/></a>
    <a href="https://github.com/huggingface/lerobot"><img alt="LeRobot" src="https://img.shields.io/badge/🤗-LeRobot-yellow"/></a>
  </div>
  
  <p>
    <a href="https://strandsagents.com/">Strands Docs</a>
    ◆ <a href="https://github.com/NVIDIA/Isaac-GR00T">NVIDIA GR00T</a>
    ◆ <a href="https://github.com/huggingface/lerobot">LeRobot</a>
    ◆ <a href="https://github.com/dusty-nv/jetson-containers">Jetson Containers</a>
  </p>
</div>

Control robots with natural language through [Strands Agents](https://github.com/strands-agents/sdk-python). Integrates [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) for vision-language-action policies and [LeRobot](https://github.com/huggingface/lerobot) for universal robot support.

## How It Works

```mermaid
graph LR
    A[Natural Language<br/>'Pick up the red block'] --> B[Strands Agent]
    B --> C[Robot Tool]
    C --> D[Policy Provider<br/>GR00T/Mock]
    C --> E[LeRobot<br/>Hardware Abstraction]
    D --> F[Action Chunk<br/>16 timesteps]
    F --> E
    E --> G[Robot Hardware<br/>SO-101/GR-1/G1]

    classDef input fill:#2ea44f,stroke:#1b7735,color:#fff
    classDef agent fill:#0969da,stroke:#044289,color:#fff
    classDef policy fill:#8250df,stroke:#5a32a3,color:#fff
    classDef hardware fill:#bf8700,stroke:#875e00,color:#fff

    class A input
    class B,C agent
    class D,F policy
    class E,G hardware
```

## Architecture

```mermaid
flowchart TB
    subgraph Agent["🤖 Strands Agent"]
        NL[Natural Language Input]
        Tools[Tool Registry]
    end

    subgraph RobotTool["🦾 Robot Tool"]
        direction TB
        RT[Robot Class]
        TM[Task Manager]
        AS[Async Executor]
    end

    subgraph Policy["🧠 Policy Layer"]
        direction TB
        PA[Policy Abstraction]
        GP[GR00T Policy]
        MP[Mock Policy]
        CP[Custom Policy]
    end

    subgraph Inference["⚡ Inference Service"]
        direction TB
        DC[Docker Container]
        ZMQ[ZMQ Server :5555]
        TRT[TensorRT Engine]
    end

    subgraph Hardware["🔧 Hardware Layer"]
        direction TB
        LR[LeRobot]
        CAM[Cameras]
        SERVO[Feetech Servos]
    end

    NL --> Tools
    Tools --> RT
    RT --> TM
    TM --> AS
    AS --> PA
    PA --> GP
    PA --> MP
    PA --> CP
    GP --> ZMQ
    ZMQ --> TRT
    TRT --> DC
    AS --> LR
    LR --> CAM
    LR --> SERVO

    classDef agentStyle fill:#0969da,stroke:#044289,color:#fff
    classDef robotStyle fill:#2ea44f,stroke:#1b7735,color:#fff
    classDef policyStyle fill:#8250df,stroke:#5a32a3,color:#fff
    classDef infraStyle fill:#bf8700,stroke:#875e00,color:#fff
    classDef hwStyle fill:#d73a49,stroke:#a72b3a,color:#fff

    class NL,Tools agentStyle
    class RT,TM,AS robotStyle
    class PA,GP,MP,CP policyStyle
    class DC,ZMQ,TRT infraStyle
    class LR,CAM,SERVO hwStyle
```

## Quick Start

```python
from strands import Agent
from strands_robots import Robot, gr00t_inference

# Create robot with cameras
robot = Robot(
    tool_name="my_arm",
    robot="so101_follower",
    cameras={
        "front": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 30}
    },
    port="/dev/ttyACM0",
    data_config="so100_dualcam"
)

# Create agent with robot tool
agent = Agent(tools=[robot, gr00t_inference])

# Start GR00T inference service
agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/model",
    port=8000,
    data_config="so100_dualcam"
)

# Control robot with natural language
agent("Use my_arm to pick up the red block using GR00T policy on port 8000")
```

## Installation

```bash
pip install strands-robots
```

From source:

```bash
git clone https://github.com/strands-labs/robots
cd robots
pip install -e .
```

<details>
<summary><b>🐳 Jetson Container Setup (Required for GR00T Inference)</b></summary>

GR00T inference requires the Isaac-GR00T Docker container on Jetson platforms:

```bash
# Clone jetson-containers
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers

# Run Isaac GR00T container (background)
jetson-containers run $(autotag isaac-gr00t) &

# Container exposes inference service on port 5555 (ZMQ) or 8000 (HTTP)
```

**Tested Hardware:**
- NVIDIA Thor Dev Kit (Jetpack 7.0)
- NVIDIA Jetson AGX Orin (Jetpack 6.x)

See [Jetson Deployment Guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/deployment_scripts/README.md) for TensorRT optimization.

</details>

## Robot Control Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as Strands Agent
    participant Robot as Robot Tool
    participant Policy as GR00T Policy
    participant HW as Hardware

    User->>Agent: "Pick up the red block"
    Agent->>Robot: execute(instruction, policy_port)
    
    loop Control Loop @ 50Hz
        Robot->>HW: get_observation()
        HW-->>Robot: {cameras, joint_states}
        Robot->>Policy: get_actions(obs, instruction)
        Policy-->>Robot: action_chunk[16]
        
        loop Action Horizon
            Robot->>HW: send_action(action)
            Note over Robot,HW: 20ms sleep (50Hz)
        end
    end
    
    Robot-->>Agent: Task completed
    Agent-->>User: "✅ Picked up red block"
```

## Tools Reference

### Robot Tool

The `Robot` class is a Strands AgentTool that provides async robot control with real-time status reporting.

| Action | Parameters | Description | Example |
|--------|------------|-------------|---------|
| `execute` | `instruction`, `policy_port`, `duration` | Blocking execution until complete | `"Pick up the cube"` |
| `start` | `instruction`, `policy_port`, `duration` | Non-blocking async start | `"Wave your arm"` |
| `status` | - | Get current task status | Check progress |
| `stop` | - | Interrupt running task | Emergency stop |

**Natural Language Examples:**

```python
# Blocking execution (waits for completion)
agent("Use my_arm to pick up the red block using GR00T policy on port 8000")

# Async execution (returns immediately)
agent("Start my_arm waving using GR00T on port 8000, then check status")

# Stop running task
agent("Stop my_arm immediately")
```

<details>
<summary><b>Robot Constructor Parameters</b></summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_name` | `str` | required | Name for this robot tool |
| `robot` | `str\|RobotConfig` | required | Robot type or config |
| `cameras` | `Dict` | `None` | Camera configuration |
| `port` | `str` | `None` | Serial port for robot |
| `data_config` | `str` | `None` | GR00T data config name |
| `control_frequency` | `float` | `50.0` | Control loop Hz |
| `action_horizon` | `int` | `8` | Actions per inference |

</details>

---

### GR00T Inference Tool

Manages GR00T policy inference services running in Docker containers.

| Action | Parameters | Description | Example |
|--------|------------|-------------|---------|
| `start` | `checkpoint_path`, `port`, `data_config` | Start inference service | `"Start GR00T on port 8000"` |
| `stop` | `port` | Stop service on port | `"Stop GR00T on port 8000"` |
| `status` | `port` | Check service status | `"Is GR00T running?"` |
| `list` | - | List all running services | `"List inference services"` |
| `find_containers` | - | Find GR00T containers | `"Find available containers"` |

**TensorRT Acceleration:**

```python
agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/model",
    port=8000,
    use_tensorrt=True,
    trt_engine_path="gr00t_engine",
    vit_dtype="fp8",    # ViT: fp16 or fp8
    llm_dtype="nvfp4",  # LLM: fp16, nvfp4, or fp8
    dit_dtype="fp8"     # DiT: fp16 or fp8
)
```

---

### Camera Tool

LeRobot-based camera management with OpenCV and RealSense support.

| Action | Parameters | Description | Example |
|--------|------------|-------------|---------|
| `discover` | - | Find all cameras | `"Discover cameras"` |
| `capture` | `camera_id`, `save_path` | Single image capture | `"Capture from /dev/video0"` |
| `capture_batch` | `camera_ids`, `async_mode` | Multi-camera capture | `"Capture from all cameras"` |
| `record` | `camera_id`, `capture_duration` | Record video | `"Record 10s video"` |
| `preview` | `camera_id`, `preview_duration` | Live preview | `"Preview camera 0"` |
| `test` | `camera_id` | Performance test | `"Test camera speed"` |

---

### Serial Tool

Low-level serial communication for Feetech servos and custom protocols.

| Action | Parameters | Description | Example |
|--------|------------|-------------|---------|
| `list_ports` | - | Discover serial ports | `"List serial ports"` |
| `feetech_position` | `port`, `motor_id`, `position` | Move servo | `"Move motor 1 to center"` |
| `feetech_ping` | `port`, `motor_id` | Ping servo | `"Ping motor 1"` |
| `send` | `port`, `data/hex_data` | Send raw data | `"Send FF FF to robot"` |
| `monitor` | `port` | Monitor serial data | `"Monitor /dev/ttyACM0"` |

---

### Teleoperation Tool

Record demonstrations for imitation learning with LeRobot.

| Action | Parameters | Description | Example |
|--------|------------|-------------|---------|
| `start` | `robot_type`, `teleop_type` | Start teleoperation | `"Start teleoperation"` |
| `stop` | `session_name` | Stop session | `"Stop recording"` |
| `list` | - | List active sessions | `"List teleop sessions"` |
| `replay` | `dataset_repo_id`, `replay_episode` | Replay episode | `"Replay episode 5"` |

---

### Pose Tool

Store, retrieve, and execute named robot poses.

| Action | Parameters | Description | Example |
|--------|------------|-------------|---------|
| `store_pose` | `pose_name` | Save current position | `"Save as 'home'"` |
| `load_pose` | `pose_name` | Move to saved pose | `"Go to home pose"` |
| `list_poses` | - | List all poses | `"List saved poses"` |
| `move_motor` | `motor_name`, `position` | Move single motor | `"Move gripper to 50%"` |
| `incremental_move` | `motor_name`, `delta` | Small movement | `"Move elbow +5°"` |
| `reset_to_home` | - | Safe home position | `"Reset to home"` |

---

## Supported Robots

| Robot | Config | Cameras | Description |
|-------|--------|---------|-------------|
| SO-100/SO-101 | `so100`, `so100_dualcam`, `so100_4cam` | 1-4 | Single arm desktop robot |
| Fourier GR-1 | `fourier_gr1_arms_only` | 1 | Bimanual humanoid arms |
| Bimanual Panda | `bimanual_panda_gripper` | 3 | Dual Franka Emika arms |
| Unitree G1 | `unitree_g1` | 1 | Humanoid robot platform |

<details>
<summary><b>GR00T Data Configurations</b></summary>

| Config | Video Keys | State Keys | Description |
|--------|------------|------------|-------------|
| `so100` | `video.webcam` | `state.single_arm`, `state.gripper` | Single camera |
| `so100_dualcam` | `video.front`, `video.wrist` | `state.single_arm`, `state.gripper` | Front + wrist |
| `so100_4cam` | `video.front`, `video.wrist`, `video.top`, `video.side` | `state.single_arm`, `state.gripper` | Quad camera |
| `fourier_gr1_arms_only` | `video.ego_view` | `state.left_arm`, `state.right_arm`, `state.left_hand`, `state.right_hand` | Humanoid arms |
| `bimanual_panda_gripper` | `video.right_wrist_view`, `video.left_wrist_view`, `video.front_view` | EEF pos/quat + gripper | Dual arm EEF |
| `unitree_g1` | `video.rs_view` | `state.left_arm`, `state.right_arm`, `state.left_hand`, `state.right_hand` | G1 humanoid |

</details>

## Policy Providers

```mermaid
classDiagram
    class Policy {
        <<abstract>>
        +get_actions(observation, instruction)
        +set_robot_state_keys(keys)
        +provider_name
    }

    class Gr00tPolicy {
        +data_config
        +policy_client: ZMQ
        +get_actions()
    }

    class MockPolicy {
        +get_actions()
        Returns random actions
    }

    class CustomPolicy {
        +get_actions()
        Your implementation
    }

    Policy <|-- Gr00tPolicy
    Policy <|-- MockPolicy
    Policy <|-- CustomPolicy
```

```python
from strands_robots import create_policy

# GR00T policy (requires inference server)
policy = create_policy(
    provider="groot",
    data_config="so100_dualcam",
    host="localhost",
    port=8000
)

# Mock policy (for testing)
policy = create_policy(provider="mock")
```

## Project Structure

```
strands-robots/
├── strands_robots/
│   ├── __init__.py              # Package exports
│   ├── robot.py                 # Universal Robot class (AgentTool)
│   ├── policies/
│   │   ├── __init__.py          # Policy ABC + factory
│   │   └── groot/
│   │       ├── __init__.py      # Gr00tPolicy implementation
│   │       ├── client.py        # ZMQ inference client
│   │       └── data_config.py   # Robot embodiment configurations
│   └── tools/
│       ├── gr00t_inference.py   # Docker service manager
│       ├── lerobot_camera.py    # Camera operations
│       ├── lerobot_calibrate.py # Calibration management
│       ├── lerobot_teleoperate.py # Recording/replay
│       ├── pose_tool.py         # Pose management
│       └── serial_tool.py       # Serial communication
├── test.py                      # Integration example
└── pyproject.toml               # Package configuration
```

## Example: Complete Workflow

```python
#!/usr/bin/env python3
from strands import Agent
from strands_robots import Robot, gr00t_inference, lerobot_camera, pose_tool

# 1. Create robot with dual cameras
robot = Robot(
    tool_name="orange_arm",
    robot="so101_follower",
    cameras={
        "wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 15},
        "front": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 15},
    },
    port="/dev/ttyACM0",
    data_config="so100_dualcam",
)

# 2. Create agent with all robot tools
agent = Agent(
    tools=[robot, gr00t_inference, lerobot_camera, pose_tool]
)

# 3. Start inference service
agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/gr00t-wave/checkpoint-300000",
    port=8000,
    data_config="so100_dualcam",
)

# 4. Interactive control loop
while True:
    user_input = input("\n🤖 > ")
    if user_input.lower() in ["exit", "quit"]:
        break
    agent(user_input)

# 5. Cleanup
agent.tool.gr00t_inference(action="stop", port=8000)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_ASSETS_DIR` | Custom directory for robot model assets (MJCF, meshes) | `~/.strands_robots/assets/` |
| `STRANDS_ROBOT_MODE` | Default mode for `Robot()` factory: `sim` / `real` / `auto` | `sim` |
| `STRANDS_TRUST_REMOTE_CODE` | Allow downloading + executing model code | `false` |
| `MUJOCO_GL` | GL backend for the MuJoCo renderer | auto |
| `STRANDS_MESH` | Set to `false` to disable Zenoh mesh networking globally | `true` |
| `STRANDS_MESH_PORT` | TCP port for the local Zenoh router | `7447` |
| `STRANDS_MESH_BACKEND` | Selects the mesh transport implementation: `zenoh` (LAN-native), `iot` (AWS IoT MQTT), or `bridge` (Zenoh + IoT cross-transport). Unknown values fall back to `zenoh` with a WARNING; the policy is to keep the mesh running rather than crash on a typo. | `zenoh` |
| `ZENOH_CONNECT` | Comma-separated list of remote Zenoh endpoints to connect to | - |
| `ZENOH_LISTEN` | Comma-separated list of endpoints for the local Zenoh listener | - |
| `STRANDS_MESH_AUDIT_DIR` | Directory for the safety audit log (`mesh_audit.jsonl`) and sequence-counter sidecar (`mesh_audit.seq.json`) | `~/.strands_robots/` |
| `STRANDS_MESH_AUDIT_MAX_BYTES` | Maximum size (bytes) of the active audit log before rotation. Hard-capped at 10 GiB. Phase-4 / E1: prevents an attacker who can publish safety events from filling the disk. | `104857600` (100 MiB) |
| `STRANDS_MESH_AUDIT_MAX_FILES` | Maximum number of rotated audit log copies kept (`mesh_audit.jsonl.1` … `.N`). Hard-capped at 100. Older rotations are discarded. | `5` |
| `STRANDS_MESH_BRIDGE_TOPICS` | Comma-separated allowlist of key-expression suffixes the Zenoh / AWS IoT bridge transport forwards across the cloud boundary. Read by `mesh.transport.bridge_transport`. Empty / unset uses the default suffix set (`presence,health,safety/event,safety/estop,cmd,response,broadcast`); high-volume telemetry (`state`, `pose`, `imu`, `odom`, `lidar`, `camera`, `input`, `hand`) is **not** bridged by default — set this var explicitly to opt in. Match semantics are exact for every entry except those listed in `STRANDS_MESH_BRIDGE_TOPICS_PREFIX`. | `presence,health,safety/event,safety/estop,cmd,response,broadcast` |
| `STRANDS_MESH_BRIDGE_TOPICS_PREFIX` | Comma-separated list of bridge filter entries that match as path-prefix (entry matches `entry/<anything>`). Default: `response` (so `response/<turn-id>` bridges). All other entries in `STRANDS_MESH_BRIDGE_TOPICS` match exactly — Phase-4 / A2 hardening that closes the prefix-bypass attack. | `response` |
| `STRANDS_MESH_AUTH_MODE` | Wire authentication mode. `mtls` (default) enables Zenoh's TLS terminator + ACL; `none` is a dev-only mode that disables both. `none` ALSO requires `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` -- without that explicit second factor `none` raises `ValueError` at config build. | `mtls` |
| `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE` | Second-factor opt-in for `STRANDS_MESH_AUTH_MODE=none`. Accepts `1` / `true` / `yes` (case-insensitive). Without it, `auth_mode=none` is refused so a typo / forgotten env / leaked CI fixture cannot silently disable wire auth. Logs ERROR (not WARNING) on every session open when active. | unset |
| `STRANDS_MESH_NAMESPACE` | Fleet namespace prefix prepended to every key-expression. Two fleets with different namespaces cannot collide on the same network. | `strands` |
| `STRANDS_MESH_MULTICAST` | `true` enables multicast scouting. Default is gossip-only, which closes the LAN-attacker-enrollment surface. | `false` |
| `STRANDS_MESH_TLS_CA` | Filesystem path to the CA bundle used to verify peer certificates. Required when `STRANDS_MESH_AUTH_MODE=mtls`. | unset |
| `STRANDS_MESH_TLS_CERT` | Filesystem path to this peer's certificate (PEM). Required when `STRANDS_MESH_AUTH_MODE=mtls`. | unset |
| `STRANDS_MESH_TLS_KEY` | Filesystem path to this peer's private key (PEM, mode 0o600). Required when `STRANDS_MESH_AUTH_MODE=mtls`. | unset |
| `STRANDS_MESH_ACL_FILE` | Filesystem path to a JSON5 ACL file enumerating each peer's exact cert CN (Zenoh 1.x does not support globs). Empty = use the permissive built-in default that allows any CA-signed peer to publish/subscribe; see `examples/mesh_acl_example.json5` for the role-separated template operators populate with their fleet's CNs. | unset |
| `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` | Required to start the mesh under `mtls` + the permissive default ACL. Without the opt-in, `Mesh.start()` logs at ERROR and returns early; `mesh.alive` stays False and no Zenoh session is acquired. Set to `1` / `true` / `yes` to acknowledge the dev/lab posture explicitly (single-tenant only). Production deployments should ship a literal-CN `STRANDS_MESH_ACL_FILE` instead. | unset |
| `STRANDS_MESH_MAX_SESSIONS` | Hard cap on simultaneous Zenoh unicast sessions. DoS bound. | `256` |
| `STRANDS_MESH_MAX_CMD_BYTES` | Per-message byte cap on `cmd` / `broadcast` topics enforced via `low_pass_filter`. | `16384` |
| `STRANDS_MESH_MAX_CAMERA_BYTES` | Per-message byte cap on camera topics. | `1048576` |
| `STRANDS_MESH_CMD_RATE_HZ` | Per-key-expression frequency cap for `cmd` topics enforced via `downsampling`. Floods are dropped at the transport before reaching the deserialiser. | `20.0` |
| `STRANDS_MESH_SAFETY_RATE_HZ` | Per-key-expression frequency cap on `safety/**` topics (`safety/estop`, `safety/resume`). Caps novel-`t` floods that bypass the receiver-side replay cache. | `2.0` |
| `STRANDS_MESH_MAX_SAFETY_BYTES` | Per-message byte cap on `safety/**` topics. Safety envelopes are small JSON dicts; jumbo-frame envelopes on this topic are DoS targeting receiver HMAC + freshness math. | `4096` |
| `STRANDS_MESH_POLICY_HOST_ALLOW` | Comma-separated host/CIDR list extending the default loopback-only `policy_host` allowlist for VLA inference targets (e.g. `vla.internal,10.0.0.0/24`). | unset |
| `STRANDS_MESH_AUDIT_PSK` | Separate PSK for HMAC-signing audit-log records. Independent of the wire PSK so audit signing can rotate on its own schedule. Unset = audit records carry no signature (`verify_audit_integrity` reports them as unverifiable). | unset |
| `STRANDS_MESH_OVERRIDE_CODE` | Operator code that clears the local emergency-stop lockout. Receivers verify the code in constant time against this env var. Unset = the local peer cannot be resumed remotely. | unset |
| `STRANDS_MESH_RESUME_FRESHNESS_S` | Maximum age (seconds) of a resume envelope before it is rejected as stale. Prevents replay of captured resume proofs outside this window. | `60` |
| `STRANDS_MESH_RESUME_FORWARD_SKEW_S` | Maximum forward clock skew (seconds) tolerated in a resume envelope timestamp. Rejects envelopes timestamped in the future beyond this tolerance. | `5` |
| `STRANDS_MESH_RESUME_REPLAY_CACHE_MAX` | Maximum number of `proof_nonce` values remembered in the per-receiver replay cache. Bounded LRU eviction prevents memory exhaustion from high-volume resume attempts. | `4096` |
| `STRANDS_MESH_DEDUP_TTL` | Bridge-transport deduplication window (seconds). Caps how long the same envelope nonce is remembered across the Zenoh + IoT subscriber wrappers. | `120` |
| `STRANDS_MESH_CAMERA_PRESIGN_TTL` | Lifetime (seconds) of presigned S3 GET URLs published in `/camera/.../ref` messages. Capped at 3600. | `60` |
| `STRANDS_MESH_CAMERA_DISABLED` | Set to `true` to disable camera publishing entirely (privacy kill switch). The camera publisher in `strands_robots.mesh` short-circuits before any frame is built; no `/camera/**` traffic is emitted. | `false` |
| `STRANDS_MESH_CAMERA_HZ` | Per-camera publish frequency (Hz) on `<ns>/<peer>/camera/<name>`. `0` (the default) disables periodic publishing -- frames are emitted on-demand only. Operators set a non-zero value to drive a fixed-rate stream. | `0` (off) |
| `STRANDS_MESH_POSE_HZ` | Pose-topic publish cadence (Hz). Drives the `<peer>/pose` SE(3) loop in `strands_robots.mesh.sensors`; only runs if the robot exposes a `pose` attribute. | `10.0` |
| `STRANDS_MESH_HEALTH_HZ` | Health-topic publish cadence (Hz). Drives the `<peer>/health` loop (battery / CPU / memory / disk / temps); only runs if the robot exposes a `health` attribute. | `0.5` |
| `STRANDS_MESH_IMU_HZ` | IMU-topic publish cadence (Hz). Drives the `<peer>/imu` loop (roll / pitch / yaw / gyro / accel); only runs if the robot exposes an `imu` attribute. | `10.0` |
| `STRANDS_MESH_ODOM_HZ` | Dead-reckoning odometry publish cadence (Hz). Drives the `<peer>/odom` loop. | `10.0` |
| `STRANDS_MESH_LIDAR_SUMMARY_HZ` | LiDAR point-cloud summary cadence (Hz). Drives the `<peer>/lidar/summary` loop. The full state topic (`lidar/state`) runs at a separate compile-time cadence. | `5.0` |
| `STRANDS_MESH_HAND_HZ` | End-effector publish cadence (Hz). Drives the `<peer>/hand/<name>/state` loop (joint positions / forces). | `50.0` |
| `STRANDS_MESH_MAP_INFO_HZ` | Map-metadata publish cadence (Hz). Drives the `<peer>/map/info` loop. | `0.2` |
| `STRANDS_MESH_FILTER_INTERFACES` | Optional comma-separated NIC allowlist for the `low_pass_filter` rules. Unset means "every link" (Zenoh's `SubjectProperty::Wildcard`). Operators set this on multi-homed hosts (e.g. WAN + LAN cap) to scope the byte caps to a specific interface. | unset (wildcard) |
| `STRANDS_MESH_CAMERA_S3_BUCKET` | S3 bucket for the IoT camera-offload path (`mesh.iot.camera_offload`). Frames are uploaded to the bucket and a presigned GET URL is published on `/camera/.../ref` instead of the raw bytes. Empty = the offload path short-circuits with a debug log; no upload is attempted. | unset |
| `STRANDS_MESH_CAMERA_S3_PREFIX` | Optional key prefix prepended to camera-offload S3 object keys. Trailing slashes are stripped. | unset |
| `STRANDS_MESH_CA_PINS` | Comma-separated additional Amazon Root CA1 SHA-256 fingerprints (64-char lowercase hex). Augments the built-in pin tuple so operators can stage a future-rotation pin ahead of a code-level rotation; the built-in tuple is always included. Invalid entries are logged at WARNING and skipped. R7-3 hardening for the CA-pin time-bomb concern. | unset |
| `STRANDS_MESH_DISABLE_CA_PIN` | Break-glass: skip the SHA-256 pin check when **downloading** `AmazonRootCA1.pem` during IoT provisioning. A WARNING is logged on every disabled run. As of round-3, this NEVER applies to the on-disk re-use path — an existing CA file is always raw-pin-checked, so a rogue CA from a prior compromised run cannot be silently re-used. To refresh a re-encoded cert behind a proxy, delete the file and re-run with the override. Should never be set in production. | `false` |
| `STRANDS_MESH_HF_REPO_ALLOW` | Comma-separated list of HuggingFace org prefixes (or full `<org>/<repo>` prefixes) that `pretrained_name_or_path` accepts in mesh `execute`/`start` commands. Defaults to `nvidia,huggingface,lerobot`. Round-3 hardening of threat-vector #3: blocks an authenticated peer from steering a robot at an attacker-controlled HF repo. | unset |
| `STRANDS_MESH_POLICY_TYPE_ALLOW` | Comma-separated list of additional `policy_type` values that mesh `execute`/`start` commands accept on top of the built-in allowlist (`mock`, `groot`, `lerobot`, `lerobot_local`, `act`, `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi0fast`, `smolvla`, `sac`). | unset |
| `GROOT_API_TOKEN` | API token for GR00T inference service | - |
| `STRANDS_ROBOT_MODE` | Override `Robot()` factory mode detection (`sim`, `real`, `auto`) | `auto` |
| `STRANDS_TRUST_REMOTE_CODE` | Set to `1` to opt into HuggingFace `trust_remote_code` for `lerobot_local` policies | unset |
| `MUJOCO_GL` | OpenGL backend for MuJoCo (`egl`, `osmesa`, `glfw`) | auto-detected |
| `STRANDS_LIBERO_ACTION_LOG` | Set to `1` to emit per-step diagnostic logs from the LIBERO OSC controller (action keys, delta scale, EEF tracking, gripper polarity, qpos/ctrl deltas). Logs the first N steps per episode. | unset |
| `STRANDS_LIBERO_ACTION_LOG_MAX` | Max number of `apply()` calls to log per episode when `STRANDS_LIBERO_ACTION_LOG=1`. | `50` |
| `STRANDS_LIBERO_STATE_LOG` | Set to `1` to emit per-step diagnostic logs of the state values (`state.x/y/z/roll/pitch/yaw/gripper`) the LIBERO adapter feeds to the GR00T policy. Pairs with `STRANDS_LIBERO_ACTION_LOG` for end-to-end interface bisection. | unset |
| `STRANDS_LIBERO_STATE_LOG_MAX` | Max number of `augment_observation()` calls to log per episode when `STRANDS_LIBERO_STATE_LOG=1`. | `50` |
| `STRANDS_GROOT_WIRE_LOG` | Path to a directory where `Gr00tPolicy` will dump pre-inference observations + post-inference action chunks as pickle files (one per `get_actions` call, named `{local,service}_call{N:04d}.pkl`). Used by the #187 bisection plan to verify whether LOCAL and SERVICE inference paths send byte-identical observations to the model. Run an eval once with each mode into the same dir, then `np.allclose` matching files. | unset |
| `STRANDS_GROOT_WIRE_LOG_MAX_CALLS` | Cap on number of wire-payload dumps per process when `STRANDS_GROOT_WIRE_LOG` is set. Prevents multi-GB pickle archives on long evals. The first few calls are enough to bisect a divergence. | `10` |

### Mesh Networking

Every `Robot()` and `Simulation()` constructed in a process is automatically a
peer on the local Zenoh mesh — no manual setup required.  Peers on the same
LAN discover each other via Zenoh multicast scouting, and a single
process-wide `zenoh.Session` is shared (ref-counted) across every robot or
simulation in the same Python process.

```python
from strands_robots import Robot
sim_a = Robot("so100")          # auto-joins the mesh as a peer
sim_b = Robot("so100")          # second peer in another process
print(sim_a.mesh.peers)         # discovers sim_b
sim_a.mesh.tell(sim_b.mesh.peer_id, "pick up the cube")
sim_a.mesh.emergency_stop()     # broadcast E-STOP, audited to disk
```

Disable globally with `STRANDS_MESH=false` or per-robot with
`Robot("so100", mesh=False)`.  Install the optional dependency with
`pip install strands-robots[mesh]`.

### Mesh security

The mesh layer relies on **Zenoh built-in security primitives** —
mTLS at the transport, key-expression ACLs above it, and per-key
rate / size caps for DoS bounds. There is no application-layer
crypto envelope: identity, fleet membership, replay protection,
and authorisation all happen *before* the deserialiser runs.

#### Authentication: mTLS (default)

`STRANDS_MESH_AUTH_MODE=mtls` (the default) wires Zenoh's
`transport/link/tls` block:

* `STRANDS_MESH_TLS_CA` — path to the CA bundle that signs every
  legitimate peer cert.
* `STRANDS_MESH_TLS_CERT` — this peer's PEM cert. Cert Common Name
  encodes the role: `robot-<id>` for robots, `op-<id>` for operators,
  `audit-<id>` for read-only observers.
* `STRANDS_MESH_TLS_KEY` — this peer's private key (mode 0o600).

Mutual TLS is mandatory; `verify_name_on_connect` is on. A peer
without a CA-signed cert fails the TLS handshake — its bytes never
reach the JSON deserialiser. `transport/link/protocols` is locked
to `["tls"]` so an attacker cannot downgrade to plain TCP.

For development on a trusted network, set `STRANDS_MESH_AUTH_MODE=none`
AND `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` to skip TLS + ACL. The
second-factor env var is required because `auth_mode=none` disables
BOTH the mTLS terminator AND the ACL block in a single env var; without
the explicit opt-in a typo / forgotten env / leaked CI fixture would
silently disable wire auth in production. The mesh logs an ERROR at
session-open time when this mode is active.

#### Authorisation: ACL on cert Common Name

The Zenoh `access_control` block sits above the mTLS terminator and
gates which peers may publish/subscribe on which key-expressions
based on the cert Common Name presented during the handshake.

**Important Zenoh 1.x constraints (verified against live sessions in
`tests/mesh/test_zenoh_transport_security.py`):**

1. `enabled: true` is required — without it the entire ACL block is
   silently disabled, which is why the loader hard-rejects any file
   that omits it.
2. `cert_common_names` matches **literal CNs only** — globs like
   `"robot-*"` and regexes match nothing. Operators with per-role
   enforcement enumerate every peer's exact CN.
3. Subject `interfaces` must be a non-empty list — leaving it unset
   makes the subject match nothing.
4. `key_exprs` see the user-side key (the namespace prefix is
   stripped before matching) — `**/cmd` is the robust glob;
   `<namespace>/*/cmd` never matches.

**Default ACL:** the built-in default ACL ships **permissive**: any peer with a CA-signed cert (already
verified at the mTLS handshake) may publish and subscribe on any
key. We chose permissive over default-deny because a hard-coded
default-deny with no enumerated CNs blocks every legitimate message
on first run — silent total outage. The mTLS handshake is the
fleet-membership gate; ACL is the third line of defence and
operators opt in explicitly.

**Role separation:** to enforce the standard `robot-*` /
`operator-*` / `audit-*` split, set `STRANDS_MESH_ACL_FILE` to the
template at `examples/mesh_acl_example.json5` and edit the
`cert_common_names` lists with each peer's exact CN. The example
ships three roles: `robot-*` may publish telemetry only;
`operator-*` may publish cmd/broadcast/safety topics; `audit-*` may
subscribe to everything but cannot publish.

**Audit attribution:** ACL drops are visible in Zenoh's own log
output (`zenoh::net::routing::interceptor::access_control`) — they
do NOT currently surface in `mesh/audit.py`. A peer with a valid CA
cert but a CN not enumerated in the ACL is dropped silently from
the operator's perspective; tail Zenoh logs to investigate.

#### Discovery: gossip-only by default

`STRANDS_MESH_MULTICAST=false` (the default) disables Zenoh's
multicast scouting. Peers find each other through gossip seeded by
explicit `ZENOH_CONNECT` endpoints. Multicast on a hostile LAN is a
discovery surface — any host that joins `224.0.0.224:7446` sees every
peer's presence broadcast. Operators on a controlled LAN can opt back
in with `STRANDS_MESH_MULTICAST=true`; we do not recommend it.

#### Fleet isolation: namespace

`STRANDS_MESH_NAMESPACE` (default `strands`) prepends an
immutable prefix to every key-expression at the routing layer. Two
fleets with different namespaces cannot route messages between each
other, even when peer-ids collide.

#### DoS bounds: downsampling + low_pass_filter

Two transport-layer caps are emitted unconditionally:

* `downsampling` enforces a per-key-expression frequency cap on
  `<ns>/*/cmd` and `<ns>/broadcast` (`STRANDS_MESH_CMD_RATE_HZ`,
  default 20 Hz). A peer publishing faster has the extra messages
  dropped at the transport — flood attacks cost nothing on the
  receiver side.
* `low_pass_filter` enforces per-message byte caps:
  `STRANDS_MESH_MAX_CMD_BYTES` (default 16 KiB) on cmd / broadcast
  topics, `STRANDS_MESH_MAX_CAMERA_BYTES` (default 1 MiB) on camera
  topics. Jumbo frames are dropped pre-deserialise.

Plus `transport/unicast/max_sessions` (`STRANDS_MESH_MAX_SESSIONS`,
default 256) caps simultaneous peer count.

#### Emergency-stop authorisation

`Mesh.emergency_stop()` and the receiver-side resume handler use the
operator override code (`STRANDS_MESH_OVERRIDE_CODE`) as a *second
factor* on top of the mTLS-bound operator role. Resume RPC carries
`HMAC(override_code, proof_nonce)`; receivers recompute it locally
and reject mismatches. Receivers without the override code configured
**fail closed** — operators must distribute the code to every peer
that should accept fleet-wide remote resume.

#### Threat-vector coverage

| Adversary tier | Coverage |
|---|---|
| LAN outsider, no cert | **Mitigated.** TLS handshake rejects the connection. |
| Cert from a different CA | **Mitigated.** Our CA bundle does not verify it; handshake fails. |
| Valid `robot-*` cert tries to publish on `*/cmd` (with `STRANDS_MESH_ACL_FILE` set) | **Mitigated.** The role-separated ACL template only allows `robot-*` peers ingress on telemetry topics; cmd publish is denied. **Without** `STRANDS_MESH_ACL_FILE`, the permissive default allows it — operators must opt in to per-role enforcement. |
| Valid `op-*` cert floods `cmd` topic at 1 kHz | **Mitigated.** `downsampling` caps ingress at the configured frequency (default 20 Hz); the rest is dropped pre-deserialise. Verified live in `test_zenoh_transport_security.py::TestDownsamplingRateCap`. |
| Valid `op-*` cert sends 100 MiB camera frame | **Mitigated.** `low_pass_filter` caps camera bytes at the configured limit (default 1 MiB) before the receiver allocates buffers. Verified live in `TestLowPassFilterByteCap`. |
| Valid `op-*` cert tries to hijack another operator's RPC turn_id | **Mitigated.** The mesh response handler requires `responder_id` to match the original target for point-to-point sends; mismatched responses are dropped. Verified in `test_application_security.py::test_p4_d1_response_hijack_rejected_for_point_to_point`. |
| Peer with valid CA cert but CN not in operator's ACL list | **Mitigated** when `STRANDS_MESH_ACL_FILE` is set with literal CN allowlists — the default-deny rule drops every put + declare_subscriber from a CN that does not appear in any subject (verified live in `TestACLEnforcement::test_unknown_cn_dropped_by_default_deny`). **Not mitigated by default** — the built-in permissive ACL admits any CA-signed peer to every key-expression. The default-state safety net is `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL`: without that opt-in, `Mesh.start()` refuses to acquire a Zenoh session under the permissive default in `mtls` mode, so an operator who forgot to ship an ACL file fails closed instead of running wide-open. Production deployments must ship a literal-CN ACL file. |
| Two fleets share a network | **Mitigated.** `STRANDS_MESH_NAMESPACE` isolates routing. |
| Stolen cert + key (host fully compromised) | **Out of scope.** The peer is the attacker. Operator response: revoke the cert at the CA, restart the fleet. |

#### Payload semantics: `validate_command`

After the wire layer authenticates and authorises a command, the
payload still goes through `mesh.security.validate_command` to bound
its contents — instruction length, duration, step counts, the
`policy_host` allowlist (loopback only by default), the HuggingFace
repo prefix gate (`STRANDS_MESH_HF_REPO_ALLOW`), and the policy
type / provider allowlist (`STRANDS_MESH_POLICY_TYPE_ALLOW` is the
single env var that extends both the `policy_type` and `policy_provider`
gates -- they share one allowlist by design). These
guard against an authorised peer requesting a 24-hour `execute` action
or steering the robot at an attacker-controlled inference server.

#### Bridge transport (Zenoh + AWS IoT)

Bridge-transport deployments still need cross-transport deduplication
because the same Zenoh + IoT MQTT topic can deliver the same payload
twice. The bridge transport in `mesh.transport.bridge_transport`
fingerprints incoming samples and dispatches each unique
(`sender_id`, `turn_id`, `command`) tuple once before forwarding to the
application-layer handler.

For a complete walkthrough of what each layer protects against, see the module docstring of `strands_robots.mesh.security`. The transport-side Zenoh session config and the ACL builder are constructed by the `strands_robots.mesh` package internals; pinning their public surface to the env vars in the matrix above keeps user-facing docs decoupled from the underscore-prefixed implementation modules.

### Cache Directory

Robot model assets (MJCF XML files and meshes) are cached in:

```
~/.strands_robots/
└── assets/           # Downloaded robot models (from robot_descriptions / MuJoCo Menagerie)
    ├── trs_so_arm100/
    ├── franka_emika_panda/
    └── ...
```

To clear the cache: `rm -rf ~/.strands_robots/assets/`

To change the cache location: `export STRANDS_ASSETS_DIR=/path/to/custom/dir`

## Simulation (MuJoCo)

`strands-robots` ships a MuJoCo-backed simulation AgentTool - 58 actions
exposed to any Strands agent for world composition, physics, policy
execution, and video/dataset recording.

### Install

```bash
pip install "strands-robots[sim-mujoco]"
# For LeRobotDataset recording (parquet + training data):
pip install "strands-robots[sim-mujoco,lerobot]"
```

### Quick start

```python
from strands_robots.simulation import Simulation

sim = Simulation(tool_name="sim", mesh=False)
sim.create_world()
sim.add_robot(name="arm", data_config="so100")
sim.add_object(name="cube", shape="box", position=[0.3, 0, 0.05])
sim.add_camera(name="topdown", position=[0, 0, 1.5], target=[0, 0, 0])

sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=200,
               control_frequency=50.0, fast_mode=True)

frame = sim.render(camera_name="topdown")  # returns {status, content:[text, image]}
```

### 58 actions grouped

- **World & objects**: `create_world`, `load_scene`, `add_robot`,
  `add_object`, `move_object`, `list_objects`, `list_robots`,
  `remove_robot`, `remove_object`, `destroy`, `reset`, `get_state`,
  `save_state`, `load_state`, `list_checkpoints`.
- **Physics**: `step`, `set_timestep`, `set_gravity`, `apply_force`,
  `raycast`, `multi_raycast`, `set_body_properties`,
  `set_geom_properties`, `get_body_state`, `get_joint_state`,
  `set_joint_positions`, `set_joint_velocities`, `forward_kinematics`,
  `get_mass_matrix`, `inverse_dynamics`, `get_total_mass`,
  `get_jacobian`, `get_energy`, `get_contacts`, `get_sensor_data`.
- **Cameras & rendering**: `add_camera`, `remove_camera`, `render`,
  `render_depth`, `render_all`, `start_cameras_recording`,
  `stop_cameras_recording`, `get_cameras_recording_status`.
- **Policy**: `start_policy`, `run_policy`, `stop_policy`,
  `replay_episode`, `eval_policy`.
- **Randomization**: `randomize`.
- **Recording (LeRobotDataset)**: `start_recording`, `stop_recording`,
  `get_recording_status`.
- **Introspection & util**: `get_features`, `list_urdfs`, `register_urdf`,
  `export_xml`, `open_viewer`, `close_viewer`.

### Common footguns

- **Planes must be static.** `add_object(shape="plane")` auto-sets
  `is_static=True`. Passing `is_static=False` on a plane is a hard error
  (MuJoCo planes are infinite and can't have dynamic mass).
- **Camera orientation.** Pass `target=[x,y,z]` to look at a point -
  without it the camera faces forward by default. `target == position`
  errors.
- **MP4 vs dataset recording.** `start_cameras_recording` writes plain
  MP4 per-camera and runs under `[sim-mujoco]` alone. `start_recording`
  writes a LeRobotDataset (parquet + MP4 + schema) and requires the
  `[lerobot]` extra.
- **Policy running → mutations blocked.** While a policy runs on any
  robot, state-mutating actions (`reset`, `set_gravity`, joint setters,
  `apply_force`, `set_body_properties`, `set_geom_properties`,
  `load_state`, `randomize`, `move_object`) error with *"Cannot 'X'
  while a policy is running."* Stop it first with
  `stop_policy(robot_name='...')`.
- **Horizon parameters.** `run_policy` accepts either `duration` +
  `control_frequency` (real-time) OR `n_steps` + `control_frequency`
  (step-count). Pass `fast_mode=True` to skip the between-step sleep
  during batch eval / data collection.
- **Name collisions.** Objects, bodies, robots, and cameras share the
  MuJoCo name table. Robot joints and actuators are auto-namespaced as
  `{robot_name}/{joint}` in multi-robot scenes. Object geoms are
  injected as `{object_name}_geom`; `set_geom_properties` accepts the
  bare object name as an alias.
- **Oversized render**: MuJoCo's offscreen framebuffer is capped by
  `<global offwidth="W" offheight="H"/>` in MJCF. Requesting a bigger
  render now errors with a plain message naming the cap - either lower
  the request or rebuild the model with larger dims.

### Self-healing features

- Unknown parameters are rejected with *"Unknown parameter X for action
  Y. Valid: [...]"* so the LLM learns the correct name without trial-
  and-error.
- Missing required parameters produce *"Action X requires parameter Y."*
  (no Python `TypeError` leaks).
- Vector dimensions and numeric dtype are validated before MuJoCo sees
  them (previously zero-length direction vectors crashed the Python
  process via `mj_ray` C-level abort).
- `destroy()` and `cleanup()` empty the renderer TLS cache and shut down
  the executor - no RSS growth across repeated create/destroy cycles.

For the full action contract and test coverage see
`tests/simulation/mujoco/test_agenttool_contract.py`.

## Contributing

We welcome contributions! Please see:
- [GitHub Issues](https://github.com/strands-labs/robots/issues) for bug reports
- [Pull Requests](https://github.com/strands-labs/robots/pulls) for contributions

## License

Apache-2.0 - see [LICENSE](LICENSE) file.

## Links

<div align="center">
  <a href="https://github.com/strands-labs/robots">GitHub</a>
  ◆ <a href="https://pypi.org/project/strands-robots/">PyPI</a>
  ◆ <a href="https://github.com/NVIDIA/Isaac-GR00T">NVIDIA GR00T</a>
  ◆ <a href="https://github.com/huggingface/lerobot">LeRobot</a>
  ◆ <a href="https://strandsagents.com/">Strands Docs</a>
</div>
