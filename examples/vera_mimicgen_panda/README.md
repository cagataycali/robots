# VERA MimicGen → Panda (real-arm IK rollout)

The flagship VERA demo on a **real arm**: a Franka Emika **Panda** in a
strands-robots `Simulation`, driven by the real VERA MimicGen policy — a WAN
video planner + AllTracker point tracker + Jacobian inverse-dynamics model that
emits 6-DoF **end-effector deltas**, converted to Panda joint targets by the
provider's **IK bridge** (auto-discovers the end-effector frame; no manual wiring).

![VERA MimicGen → Panda](../../docs/assets/vera/mimicgen_panda.gif)

```
WAN dream ─▶ AllTracker ─▶ Jacobian IDM ─▶ eef-delta (7-D) ─▶ [VERA IK bridge] ─▶ Panda joints ─▶ MuJoCo
```

## Run

```bash
# checkpoints: VERA Wave-1 + the frozen WAN 2.1 base
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

# server (bundles AllTracker + offline ckpt resolver; serves ws on :8800)
docker build -f strands_robots/policies/vera/docker/Dockerfile -t strands-vera-server:latest .
docker run --rm --gpus all --ipc=host -p 8800:8800 \
    -v "$PWD/vera-ckpts":/ckpts:ro -v "$PWD/Wan2.1-T2V-1.3B":/wan:ro \
    -e VERA_EMBODIMENT=mimicgen -e USE_OFFLINE_RESOLVE=1 \
    strands-vera-server:latest

# rollout (host needs sim-mujoco + mink + the ws client)
uv pip install -e '.[sim-mujoco]' mink websockets msgpack
MUJOCO_GL=egl python examples/vera_mimicgen_panda/rollout.py \
    --record examples/vera_mimicgen_panda/artifacts/mimicgen_panda.mp4
```

## How the IK works

VERA's `mimicgen` embodiment has `action_space="eef_delta"` (7-D: 3 translation
+ 3 axis-angle rotation + 1 gripper). The simulator drives **joint** actuators,
so the provider:

1. On `run_policy`, the MuJoCo engine's `bind_policy_sim_context` hook hands the
   provider the compiled `MjModel` + the robot's namespace.
2. The provider **auto-discovers the end-effector frame** (`ee_frame.py`: site →
   hand/tool body → kinematic leaf) and builds a [mink](https://github.com/kevinzakka/mink)
   differential-IK bridge (`sim_ik.py`).
3. Each per-step eef-delta is re-anchored on the arm's **achieved** EE pose
   (closed-loop, bounded tracking error) and solved to a full-`nq` qpos; the
   arm joints are read back by qpos address (robust to free-body DOFs).
4. The gripper column is binarized and routed to the Panda's finger joints.

> **Note.** This demo uses VERA's `mimicgen` *embodiment* (a config string),
> not the NVlabs MimicGen package -- which has no PyPI release. The
> `mimicgen` name on PyPI is unaffiliated (a dependency-confusion risk) and
> is intentionally not a dependency. If you need NVlabs MimicGen for data
> generation, install it from source:
>
> ```bash
> pip install "mimicgen @ git+https://github.com/NVlabs/mimicgen.git"
> ```

## See also

- Provider docs: [`docs/policies/vera.md`](../../docs/policies/vera.md)
