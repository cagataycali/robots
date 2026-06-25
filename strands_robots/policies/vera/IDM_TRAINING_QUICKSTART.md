# Adding a new robot to VERA — IDM training quickstart (closing the loop)

Goal: take a robot we support in `strands-robots` that VERA does **not** yet ship
an IDM for, train its small **Jacobian IDM**, and serve it through our existing
VERA provider — **zero provider code changes** (the transport + action-binding +
auto-IK are already embodiment-general).

> Recall the asymmetry (see `PAPER_CLAIM_RECONCILED.md`): the **video planner is
> shared** (train once / reuse OMNI), and **only the IDM is per-robot** — a small,
> data-efficient model (frozen VGGT backbone + flow→action head). This is the only
> piece you train to onboard a new embodiment.

---

## The 4 steps

```
 (1) collect self-play data  ->  (2) pack dataset  ->  (3) train IDM  ->  (4) serve + drive
        (our sim recorder)         (vera datasets)      (vera.main)       (our VeraPolicy)
```

### 1. Collect self-play / teleop data for the robot
You need ``(rgb views, robot du-action)`` pairs — the IDM regresses the local
action↔optical-flow map. Use whichever we already have:

- **Sim self-play / scripted rollouts** via `strands-robots` simulation +
  `LeRobotDataset` recording (the 60+ action sim tool records frames + actions):
  ```python
  from strands_robots import Robot
  robot = Robot("ur5e")                       # any arm in our registry
  robot(action="record_dataset", repo_id="me/ur5e_selfplay",
        policy="mock", episodes=200, cameras=["front","wrist"])
  ```
- **Real teleop** via the LeRobot path (`mode="real"`, `record_dataset`).

What VERA's IDM dataset needs per step (see `vera/datasets/core/actions.py`):
`rgb [T,V,3,H,W]`, and the **per-embodiment ``du`` action** (the local action
encoding — eef-delta / joint-delta). Match the ``action_space`` you intend to
serve (eef_delta is the most transferable; joint_position is embodiment-exact).

### 2. Pack the dataset into VERA's self-contained core
VERA's loader (`vera/datasets/core/`) reads packed JPEG/qint8 or decord video.
Point it at your recorded episodes via ``VERA_DATA_PREFIX`` and a dataset config
mirroring the shipped ones:

```bash
export VERA_DATA_PREFIX=/data/vera                 # holds your packed episodes
# Author vera/configurations/dataset/<your_robot>.yaml by copying
#   dataset/mimicgen_packed_v3.yaml (eef) or dataset/pusht_packed.yaml,
# editing: views, action_dim, du action model (actions.py), fps/aspect.
```

### 3. Train the Jacobian IDM (the only per-robot training)
Copy a shipped IDM config and point it at your dataset. The backbone (VGGT/DINO)
is **frozen** — you train the small decoder + action head, so this is fast and
data-efficient:

```bash
# inside the VERA env (torch 2.6 / the container or a vera venv):
pip install -e ".[idm,video,eval]"

# eef-delta arm (copy the mimicgen recipe):
python -m vera.main \
  --config-name=config_jacobian_mimicgen_vggt_v3_taskbalanced \
  dataset=<your_robot> \
  experiment.training.batch_size=8 \
  wandb.mode=offline                      # offline-safe; we resolve ckpts locally

# joint-space (copy the pusht/allegro recipe pattern):
python -m vera.main --config-name=config_pusht_vggt_fusion_jacobian dataset=<your_robot>
```

Output: a ``model.ckpt`` + a ``config.yaml`` sidecar (the IDM's wire metadata:
``action_dim``, ``view_keys``, ``gripper_dim_index`` …). Drop them into your
checkpoint root next to the released ones, with a ``provenance.json`` carrying the
``wandb_run`` so the container resolves it offline:

```
vera-ckpts/
  idm-<your_robot>-<runid>/
    model.ckpt
    config.yaml
    provenance.json   # {"wandb_run": "entity/project/<runid>", ...}
```

> The container's `wandb_offline_resolve.py` indexes any ``provenance.json`` under
> ``/ckpts`` → your new IDM is found by run-id with **no wandb network**.

### 4. Serve it + drive any of our robots (no provider changes)
Reuse the WAN/DFoT planner you already have; only the IDM is new. Point the
server at your IDM run-id and launch via our docker runtime:

```bash
export VERA_CKPT_ROOT=/abs/vera-ckpts
export VERA_WAN_CKPT_ROOT=/abs/Wan2.1-T2V-1.3B    # eef/omni planner base
export VERA_DYNAMICS_RUN_ID=<your_runid>          # your new IDM
VERA_EMBODIMENT=mimicgen docker compose \
  -f strands_robots/policies/vera/docker/docker-compose.yml up
```

Then drive it from `strands-robots` — **the provider auto-binds actions and
auto-configures IK from the robot's MuJoCo model** (zero config):

```python
from strands_robots import Robot
from strands_robots.policies import create_policy

robot = Robot("ur5e")                              # the arm you trained the IDM for
policy = create_policy("vera", embodiment="mimicgen", server_mode="docker",
                       auto_launch_server=False)
# No set_ik_target needed: the sim calls policy.set_sim_context(mj_model, ns)
# which auto-discovers the ee-frame (attachment_site / hand / leaf body) and
# wires the Cartesian IK. joint_position embodiments skip IK entirely.
robot(action="run_policy", policy=policy, instruction="pick up the red cube", n_steps=200)
```

---

## What "closing the loop" gives you

| Piece | Status |
|---|---|
| Transport / protocol | embodiment-agnostic ✅ |
| Action binding (joint / eef / cartesian) | generic ✅ |
| **IK end-effector frame** | **auto-discovered from the MjModel — zero config ✅** |
| Offline IDM checkpoint resolution | provenance.json index ✅ |
| New-robot onboarding | **train one small IDM** (this doc) → drop ckpt → serve |

So onboarding a new embodiment is now: **collect data → train a small IDM → drop
the checkpoint**. The runtime, action-binding, and IK are already general — you
never touch provider code.

---

## Tips / gotchas
- **Prefer eef_delta** for cross-arm reuse: Cartesian deltas + our IK transfer
  across kinematically-similar arms; joint_position is exact-embodiment only.
- **ee-frame auto-discovery** prefers a TCP **site** (``attachment_site`` etc.),
  then a hand/gripper **body**, then the kinematic **leaf body**. If your robot's
  XML lacks a sensible tool frame, add an ``attachment_site`` to its MJCF or pass
  ``policy.set_ik_target(mj_model, ee_frame_name=...)`` explicitly.
- **Validate transfer before trusting it:** run the IDM on the target arm and
  watch the IK tracking error (the provider logs ``mean_mm`` / ``max_mm`` per
  chunk at DEBUG). Large tracking error => retrain the IDM on that arm's data.
- **Gripper:** ensure your robot's gripper joint name contains ``gripper`` /
  ``finger`` so the provider routes the binarized gripper column to it.
