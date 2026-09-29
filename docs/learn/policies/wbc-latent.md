---
description: wbc_latent decodes a VLA's SONIC motion tokens into Unitree G1 joint targets with NVIDIA's decoder and tracks them with SONIC's PD law.
---

# wbc_latent

By the end of this page you can run a pi0.5 checkpoint that predicts SONIC motion tokens on a simulated Unitree G1, pick its decoder variant, and know why tokens never touch the joints.

```bash
pip install 'strands-robots[wbc,lerobot]'    # onnxruntime for the decoder, lerobot for pi0.5; no weights bundled
```

## What it is

The recipe in "Bringing Humanoids to LeRobot" fine-tunes pi0.5 to predict a 66-wide action at 50 fps: 64 SONIC latent motion tokens and two gripper commands (`nepyope/pi05-can-to-martino-12k`, trained on `nepyope/can_clean_final`). On the robot, NVIDIA's SONIC decoder (`nvidia/GEAR-SONIC`, `model_decoder.onnx`) reads one token plus the last ten frames of joint positions, velocities, base gyro, gravity direction and its own previous outputs, and emits 29 joint-position offsets that a per-joint PD law tracks.

`WBCLatentPolicy` is that stage. It wraps the token-emitting VLA (any policy whose action dicts carry `motion_token_0..63`, normally [`lerobot_local`](lerobot-local.md) with the `unitree_g1_sonic` embodiment) and one `SonicDecoder`. Its `execution_horizon` is one: the runner calls it every 50 Hz tick with fresh proprioception, it re-queries the VLA every `replan_every` ticks (20, the 2.5 Hz of NVIDIA's own client) or when its token cache is used up, decodes one token, and returns the 29 joint targets in hardware order plus `left_gripper` and `right_gripper`. The Menagerie G1 has no grippers, so the sim reports those two channels and a hardware driver maps them.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy(
    "wbc_latent",
    inner_provider="lerobot_local",
    inner_config={"pretrained_name_or_path": "nepyope/pi05-can-to-martino-12k", "policy_type": "pi05"},
    variant="default",          # the SONIC decoder whose encoder produced the training tokens
)
```

## Constructor keywords

{{providers:kwargs:wbc_latent}}

`inner` is a built policy; `inner_provider` and `inner_config` build one for you, and for `lerobot_local` the `embodiment` defaults to `unitree_g1_sonic`, whose 66 action names keep every token (without it the 66-wide action is aligned to the 31 state keys and tokens 31 to 63 are dropped). `checkpoint` is a local `.onnx`, a directory, or a HuggingFace repo id; the default fetches one file from `nvidia/GEAR-SONIC` into the HF cache. `variant` is `default`, `low_latency` or `sonic_v1_1`; a token decodes correctly only through the decoder of the encoder that produced it, and the blog does not name its encoder, so the knob is exposed rather than guessed. A robot whose state keys are not the 29 G1 joint names is refused at `set_robot_state_keys`.

## Run it

`run_policy` on MuJoCo detects a `WBCLatentPolicy` anywhere in the policy tree and installs `WBCLatentTorqueController`: the stock position servos (`kp = 500`) are flipped to torque and every joint is tracked with `tau = kp (target - q) - kd dq` using SONIC's armature-derived gains (14 to 99 N m/rad) at the decoder's training clock, a 0.005 s physics step and four steps per tick. Without the shim the servos are 5x to 35x stiffer than the network expects and the robot falls. Pass `wbc_install_torque_control=False` to opt out on a torque-actuated scene.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("g1")
for name, body in (("ego_view", "g1/torso_link"), ("left_wrist", "g1/left_wrist_yaw_link"), ("right_wrist", "g1/right_wrist_yaw_link")):
    sim.add_camera(name=name, parent_body=body, position=[0.08, 0.0, 0.05], target=[0.6, 0.0, -0.2])
result = sim.run_policy(
    robot_name="g1",
    policy_object=policy,
    instruction="Bring the can to the white table",
    duration=10.0,
    control_frequency=50.0,
)
print(result["status"])
```

The camera names match the checkpoint's three image features through the embodiment's `obs_rename`. Run at 50 Hz: one token per 20 ms.

## What was measured

With the `default` decoder and NVIDIA's published standing token, the shim keeps the G1 upright for 10 s in MuJoCo: pelvis 0.79 m to 0.787 m, every target inside the joint limits, peak torque 21.6 N m, decoder 0.6 ms per tick on a laptop CPU. The same token through `low_latency` falls within two seconds: the variant coupling above.

## Licence

`nvidia/GEAR-SONIC` is dual licensed: its source is Apache-2.0 and its weights are under the NVIDIA Open Model License, which permits runtime download and asks for the notice "Licensed by NVIDIA Corporation under the NVIDIA Open Model License", logged once when the decoder loads. The pi0.5 fine-tune carries no licence field on the Hub; its base `lerobot/pi05_base` is Apache-2.0.

## Why not `wbc` or a composite

[`wbc`](wbc.md) runs the Balance and Walk controllers from a `[vx, vy, omega]` command and refuses `nvidia/GEAR-SONIC`, which ships the decoder, not those controllers. A [`CompositePolicy`](../../reference/api/policies.md#built-in-policies) merges two children per tick with no state, while the decoder must see proprioception measured after the previous target.
