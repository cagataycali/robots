# API Reference

Core classes and functions in Strands Robots.

---

## Robot (Factory)

The main entry point. Auto-detects sim vs real and returns the appropriate backend.

```python
from strands_robots import Robot

robot = Robot(
    name="so100",              # Robot name from registry
    mode="auto",               # "auto", "sim", or "real"
    backend="mujoco",          # "mujoco", "newton", or "isaac" (sim only)
    cameras=None,              # Camera config (real hardware only)
    position=None,             # [x, y, z] position in sim
    num_envs=1,                # Parallel envs (newton/isaac only)
    mesh=True,                 # Enable Zenoh P2P mesh
    **kwargs,                  # Backend-specific options
)
```

**Returns**: `MujocoBackend`, `NewtonBackend`, `IsaacSimBackend`, or `HardwareRobot` depending on mode and backend.

### Sim mode (MujocoBackend)

| Method | Returns | Description |
|---|---|---|
| `get_observation()` | `dict` | Joint positions, velocities, camera images |
| `apply_action(action)` | `dict` | Send joint targets, advance physics |
| `reset()` | `dict` | Reset to initial state |
| `render(path)` | `bytes` | Render frame as PNG |
| `add_object(...)` | `dict` | Add objects to scene |
| `run_policy(...)` | `dict` | Execute a policy in sim |

### Real mode (HardwareRobot)

The `HardwareRobot` is a Strands `AgentTool` with these actions:

| Action | Description |
|---|---|
| `execute` | Run instruction with policy (blocking) |
| `start` | Start async task execution |
| `status` | Get current task status |
| `stop` | Stop running task |
| `record` | Execute + record to LeRobotDataset |
| `replay` | Replay a recorded episode on hardware |
| `features` | Show robot observation/action features |

---

## list_robots

```python
from strands_robots import list_robots

robots = list_robots(mode="all")  # "all", "sim", "real", "both"
# Returns: [{"name": "so100", "description": "...", "category": "arm", ...}, ...]
```

---

## create_policy

```python
from strands_robots import create_policy

# By provider name
policy = create_policy("mock")
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/act_aloha_sim")
policy = create_policy("groot", port=5555, host="jetson")

# Auto-resolved from string
policy = create_policy("lerobot/act_aloha_sim_transfer_cube_human")  # → lerobot_local
policy = create_policy("zmq://jetson:5555")                          # → groot
policy = create_policy("localhost:8080")                              # → lerobot_async
```

### Policy ABC

All policies implement:

| Method | Returns | Description |
|---|---|---|
| `get_actions(obs, instruction)` | `list[dict]` | Predict actions from observation |
| `get_actions_sync(obs, instruction)` | `list[dict]` | Synchronous wrapper |
| `set_robot_state_keys(keys)` | `None` | Configure state keys |
| `provider_name` | `str` | Provider identifier |

### Providers

| Provider | Module | Description |
|---|---|---|
| `mock` | `strands_robots.policies` | Sinusoidal test actions |
| `groot` | `strands_robots.policies.groot` | NVIDIA GR00T N1.5/N1.6 via ZMQ |
| `lerobot_local` | `strands_robots.policies.lerobot_local` | Direct HuggingFace inference (ACT, Pi0, Pi0-FAST, SmolVLA, Wall-X, X-VLA, SARM, Diffusion) |
| `lerobot_async` | `strands_robots.policies.lerobot_async` | gRPC to LeRobot PolicyServer (requires `grpcio`) |
| `cosmos_predict` | `strands_robots.policies.cosmos_predict` | NVIDIA Cosmos world model |
| `gear_sonic` | `strands_robots.policies.gear_sonic` | GEAR-SONIC humanoid (135Hz ONNX) |
| `dreamgen` | `strands_robots.policies.dreamgen` | GR00T-Dreams IDM + VLA |
| `dreamzero` | `strands_robots.policies.dreamzero` | Zero-shot world action model |

---

## register_policy

```python
from strands_robots.policies import register_policy, Policy

class MyPolicy(Policy):
    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [{"joint_0": 0.1, "joint_1": -0.2}]

    def set_robot_state_keys(self, keys):
        self.keys = keys

    @property
    def provider_name(self):
        return "my_policy"

register_policy("my_policy", lambda: MyPolicy, aliases=["custom"])
```

---

## create_trainer

```python
from strands_robots import create_trainer

trainer = create_trainer(
    provider="lerobot",             # Provider name
    policy_type="act",              # Architecture
    dataset_repo_id="lerobot/so100_wipe",
    **kwargs,                       # Training hyperparameters
)
trainer.train()
```

### Trainer Providers

| Provider | Class | Description |
|---|---|---|
| `lerobot` | `LerobotTrainer` | ACT, Pi0, SmolVLA, Diffusion Policy |
| `groot` | `Gr00tTrainer` | GR00T N1.6 fine-tuning (Isaac-GR00T) |
| `dreamgen_idm` | `DreamgenIdmTrainer` | DreamGen inverse dynamics model |
| `dreamgen_vla` | `DreamgenVlaTrainer` | DreamGen VLA fine-tuning |
| `cosmos_predict` | `CosmosTrainer` | Cosmos Predict 2.5 post-training |
| `cosmos_transfer` | `CosmosTransferTrainer` | Cosmos Transfer 2.5 ControlNet (sim→real) |

### TrainConfig

Common training parameters extracted from kwargs automatically:

```python
trainer = create_trainer("lerobot",
    policy_type="act",
    dataset_repo_id="lerobot/so100_wipe",
    max_steps=10000,
    batch_size=32,
    learning_rate=1e-4,
    output_dir="./outputs",
)
```

---

## DreamGenPipeline

```python
from strands_robots import DreamGenPipeline

pipeline = DreamGenPipeline(
    video_model="wan2.1",
    idm_checkpoint="nvidia/gr00t-idm-so100",
    embodiment_tag="so100",
)

results = pipeline.run_full_pipeline(
    robot_dataset_path="/data/pick_and_place",
    instructions=["pour water", "fold towel"],
    num_per_prompt=50,
)
```

---

## CosmosTransferPipeline

```python
from strands_robots import CosmosTransferPipeline

pipeline = CosmosTransferPipeline(checkpoint_dir="/data/cosmos-transfer2-2.5")
pipeline.transfer(
    input_video="mujoco_recording.mp4",
    output_video="photorealistic.mp4",
    control_inputs=["depth"],
)
```

---

## Tools

All tools follow the Strands tool interface. Pass them to `Agent(tools=[...])`:

| Import | Tool | Description |
|---|---|---|
| `from strands_robots import Robot` | Robot control | Sim/real unified interface |
| `from strands_robots import gr00t_inference` | GR00T service | Manage GR00T Docker inference |
| `from strands_robots import lerobot_camera` | Camera ops | Discover, capture, record, preview |
| `from strands_robots import teleoperator` | Demonstration | Record demonstrations for training |
| `from strands_robots import lerobot_dataset` | Dataset mgmt | Create/manage LeRobot datasets |
| `from strands_robots import pose_tool` | Pose save/load | Store, load, execute named poses |
| `from strands_robots import serial_tool` | Servo comms | Low-level Feetech servo control |
| `from strands_robots import newton_sim` | Newton GPU | Newton GPU simulation control |
| `from strands_robots import stream` | Telemetry | Real-time telemetry streaming |
| `from strands_robots import stereo_depth` | Depth | Stereo depth estimation |
| `from strands_robots import robot_mesh` | P2P mesh | Zenoh peer-to-peer robot networking |

---

## Zenoh Robot Mesh

```python
from strands_robots import Mesh, get_peers

# Every Robot() auto-joins the mesh when mesh=True (default)
robot = Robot("so100")  # Auto-joins mesh

# Query peers
peers = get_peers()
```

---

## Optional Classes

| Class | Import | Description |
|---|---|---|
| `MujocoBackend` | `from strands_robots import MujocoBackend` | MuJoCo simulation |
| `NewtonBackend` | `from strands_robots import NewtonBackend` | Newton GPU physics |
| `IsaacSimBackend` | `from strands_robots.isaac import IsaacSimBackend` | Isaac Sim |
| `StrandsSimEnv` | `from strands_robots import StrandsSimEnv` | Gymnasium wrapper |
| `DatasetRecorder` | `from strands_robots import DatasetRecorder` | LeRobotDataset bridge |
| `RecordSession` | `from strands_robots import RecordSession` | Recording pipeline |
| `MotionLibrary` | `from strands_robots import MotionLibrary` | Reusable motions |
| `TelemetryStream` | `from strands_robots import TelemetryStream` | Streaming observability |
