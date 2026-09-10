# Microduck — locomotion policies for the Pollen 14-DOF biped

`MicroduckPolicy` wraps one of Pollen Robotics' shipped **Microduck** ONNX
policies (`alpha_walking`, `alpha_stand`, `alpha_sitstand`, `roulade`,
`ball_kick_left`/`ball_kick_right`, `roller`/`roller_crouch`,
`alpha_ground_pick`) and drives the open 14-DOF biped through the standard
`Robot(...).run_policy` seam — in MuJoCo or on hardware.

Each export is an actor with its input **normaliser fused into the graph**, so
the provider feeds the observation **raw** and never re-normalises. The policy
also **self-configures from the ONNX metadata**: `joint_names`,
`default_joint_pos`, `action_scale` and `command_names` are read from the file's
`custom_metadata_map` on first inference, so pointing it at a different weight
file reconfigures it. Explicit constructor arguments always win.

## Walking in MuJoCo

![Microduck walking in MuJoCo](../assets/microduck/microduck_walk.gif){ width=480 }

_`alpha_walking.onnx` driven forward at `vx=0.3 m/s`, filmed with a
body-tracking chase camera. Reproduce with
[`examples/microduck/render_video.py`](https://github.com/strands-labs/robots/blob/main/examples/microduck/render_video.py):_

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib  # macOS: Homebrew ffmpeg
python examples/microduck/render_video.py \
    --onnx ../microduck/policies/alpha_walking.onnx \
    --vx 0.3 --duration 8 --out walk_forward.mp4 \
    --gif docs/assets/microduck/microduck_walk.gif
```

`--vx`/`--vy`/`--vyaw` set the twist command, and any weight that runs on the
default scene (`alpha_stand`, `roulade`, `alpha_sitstand`, `alpha_ground_pick`)
drops straight in. The four skills that need a different scene — `roller`,
`roller_crouch`, `ball_kick_left`, `ball_kick_right` — need it named, see
[Skill scenes](#skill-scenes).

## Install

```bash
pip install "strands-robots[microduck]"
```

That pulls `onnxruntime` (runs the graph). Weights are not bundled — they ship
in Pollen's `microduck` repository under `policies/*.onnx`. A MuJoCo rollout
additionally needs `strands-robots[sim-mujoco]`.

## Walk in simulation

```python
from strands_robots import Robot
from strands_robots.policies.microduck import MicroduckPolicy

sim = Robot("microduck")
sim.reset()

policy = MicroduckPolicy(onnx_path="alpha_walking.onnx")
sim.run_policy(
    policy_object=policy,
    control_frequency=50,
    duration=8.0,
    policy_kwargs={"target_velocity": [0.3, 0.0, 0.0]},  # forward twist
)
```

See [`examples/microduck/microduck_walk_sim.py`](https://github.com/strands-labs/robots/blob/main/examples/microduck/microduck_walk_sim.py)
for the runnable script.

## Skill scenes

A shipped weight and the scene it was trained in are one pair. `Robot("microduck")`
resolves the entry's declared asset — flat ground, no props; four skills need
more, and Pollen ships their scenes in the same asset directory:

| skill | scene | what the scene adds |
| --- | --- | --- |
| `alpha_walking`, `alpha_stand`, `alpha_sitstand`, `roulade`, `alpha_ground_pick` | `scene.xml` (the entry's declared asset) | nothing - flat ground |
| `roller`, `roller_crouch` | `scene_rollers.xml` | four passive ankle wheels, so the feet can roll |
| `ball_kick_left`, `ball_kick_right` | `scene_ball.xml` | a 70 mm, 15 g ball placed in front of the duck |

Reach a non-default scene by path; the registry entry deliberately names the
fourteen-hinge model the catalog documents, so variants load as an explicit
asset:

```python
from pathlib import Path

from strands_robots import Robot
from strands_robots.policies.microduck import MicroduckPolicy
from strands_robots.utils import get_search_paths

scene = next(
    candidate
    for root in get_search_paths()
    if (candidate := Path(root) / "microduck" / "scene_rollers.xml").exists()
)
sim = Robot("microduck", urdf_path=str(scene))
sim.reset()
sim.run_policy(policy_object=MicroduckPolicy(onnx_path="roller.onnx"), duration=8.0)
```

Running a skill on the wrong scene is not an error: a roller policy with no
wheels simply stands, a ball-kick policy swings at nothing, and both report
success. The scene is the caller's to choose.

### The ball scene carries the ball, not the kick geometry

`scene_ball.xml` declares the ball 0.3 m straight ahead (`ball.xml`:
`pos="0.3 0 0.035"`). Pollen's training reset placed it 0.09 m ahead and
0.042 m to the side of the kicking foot, in the robot's yaw frame — 3.3x closer
and offset to the foot that swings. Driven from the shipped position,
`ball_kick_left` reports success while no robot geom comes closer than 0.109 m
to the ball centre (radius 0.035 m); the ball's low rolling resistance still
carries it forward, so the miss reads as a weak kick. For the trained geometry,
teleport the ball before the rollout as Pollen's runtime does: write the ball
free joint's `qpos` to that offset rotated into the trunk's yaw frame, zero its
`qvel`, step. The file names the joint `ball_free`, but `add_robot(name=...)`
prefixes every joint with the caller's name, so resolve the name rather than
assuming either spelling.

### Reading joint positions on the rollers scene

`scene_ball.xml` appends the ball's free joint after the robot's, so the robot's
`qpos` layout is the default one. `scene_rollers.xml` inserts two passive wheel
joints after each ankle, which moves nine of the fourteen actuated joints to a
different `qpos` index — a flat `qpos[7:21]` read gets the two left wheels where
`neck_pitch` and `head_pitch` sit on the default scene. The actuator order is
identical across all three scenes and `MicroduckPolicy` reads by joint name, so
only a raw position read has to care.

### The stance every weight was trained in

Every shipped weight bakes its start stance into the ONNX metadata as
`default_joint_pos` (all nine declare the same fourteen values); actions decode
as `motor_target = default_pose + raw_action * action_scale`, so the stance is
the origin the network's output is measured from. The same values ship as
`strands_robots.policies.microduck.MICRODUCK_DEFAULT_POSE`. The asset carries
the stance as the `STAND` keyframe of `scene.xml` and `scene_rollers.xml`; name
it at spawn and every `reset()` restores it:

```python
sim = Robot("microduck", urdf_path=str(scene), keyframe="STAND")
```

Spawning without `keyframe` is not an error: the robot starts at the zero
configuration, 0.458 rad from the trained stance at the widest joint. The live
keyframe is `STAND` although the asset's comment calls the values "STAND2"
(`keyframe="STAND2"` is refused naming the declared keyframes), and
`scene_ball.xml` declares no keyframe at all — seat the stance there yourself
from `MICRODUCK_DEFAULT_POSE`, not copied numbers.

## The observation contract

The vector is a fixed float32 concatenation (measured off Pollen's reference
`infer_policy.py` and each ONNX's `observation_names` metadata):

| block | width | source |
| --- | --- | --- |
| `base_ang_vel` | 3 | IMU angular velocity |
| `projected_gravity` | 3 | world `-Z` rotated into the base frame from `base_quat` |
| `joint_pos` | 14 | current joint position − `DEFAULT_POSE`, contract order |
| `joint_vel` | 14 | joint velocity, contract order |
| `last_action` | 14 | the **previous raw** ONNX output (not the motor target) |
| `command` | C | unified command (`twist(3) + head_pose(4) + body_pose(6)`) |

Total width is `48 + C`: **61** for the shipped alpha policies (C = 13) and 51
for legacy twist-only policies (C = 3). The width is read from `command_names`,
never hardcoded, and unused command slots stay present and zero so one layout
serves every skill. `action_scale` — explicit or read from the ONNX metadata —
must be a positive finite number: `0` would make every target exactly
`DEFAULT_POSE` while the rollout reports success, and a non-finite one would
make all fourteen targets `nan`.

## Commanding motion

The command vector defaults to all-zero (stand in place). Steer with the
well-known `target_velocity` kwarg (writes the twist slots) or replace it
wholesale with `command=`:

```python
await policy.get_actions(obs, "", target_velocity=[0.3, 0.0, 0.2])  # vx, vy, ω
```

`target_velocity` takes three components or two (`[vx, vy]`, leaving `omega` as
it was — the command vector persists across ticks); any other count, or a
non-finite component, is refused before it reaches the command, and `command=`
must be `command_names`-wide and finite. What the twist slots MEAN is a property
of the weights: the locomotion exports (`alpha_walking`, `alpha_stand`, the
`roller*` pair) read them as a velocity, but `alpha_ground_pick` reads the same
three slots as a progress encoding through a one-shot motion, and the ONNX
metadata does not distinguish the two. For those, supply the slots through
`command=` and advance them yourself.

## Hot-swapping skills

`MicroduckPolicyBundle` holds several `MicroduckPolicy` instances warm and
delegates each tick to the active one, so a controller can switch skill
mid-rollout without rebuilding sessions:

```python
from strands_robots.policies.microduck import MicroduckPolicy, MicroduckPolicyBundle

bundle = MicroduckPolicyBundle(
    {
        "walk": MicroduckPolicy(onnx_path="alpha_walking.onnx"),
        "stand": MicroduckPolicy(onnx_path="alpha_stand.onnx"),
    },
    active="stand",
    switch_on_velocity=0.1,  # auto walk<->stand by |twist|
)
```

Select explicitly with `get_actions(..., select="walk")` or `bundle.switch(...)`.
The gate arbitrates only *between* `move_key` and `idle_key` (defaults `"walk"` /
`"stand"`) and leaves any other explicit selection alone — which is what lets
`alpha_sitstand`, whose `twist[0]` is a posture flag, be selected at all.
`switch_on_velocity` must be positive and finite (omit it to switch only
explicitly); a `target_velocity` the active skill would refuse is refused before
the gate reads it, so the selection never sticks on a `nan` magnitude; and with
the gate on both keys must name held skills — a bundle keyed by weight names and
left on the default keys constructs, validates, and never switches.

## Byte-compatibility

`MicroduckPolicy.infer_raw(obs_vector)` runs the graph on a raw observation with
no normalisation — exactly as Pollen's reference deployment does. The provider's
test suite pins that an identical 61-D observation yields an action byte-identical
(0.0 max abs delta) to a bare `onnxruntime` session, and that a real MuJoCo
rollout moves the joints.
