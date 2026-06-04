# Cosmos 3 Policy Provider — Implementation Plan (strands-robots)

> Branch: `feat/cosmos3-policy` (fork: `cagataycali/robots`)
> Goal: Add **NVIDIA Cosmos 3** as a first-class VLA **policy provider** in
> `strands_robots.policies`, so `create_policy("cosmos3", ...)` returns a
> `Policy` that maps robot observations → action chunks using the
> **Cosmos3-Nano-Policy-DROID** vision-language robot policy.
> Constraint: **No NIM. Local compute only** (1× L40S, 46 GB, driver 580 → CUDA 13.0).
> Run the model locally first as a plain Python script, verify the I/O, then
> wire it into the policy contract.

---

## 0. Why Cosmos 3 = a robots Policy

Cosmos 3 is an omnimodal world model with two surfaces:

| Surface       | Inputs                       | Outputs                  |
|---------------|------------------------------|--------------------------|
| **Reasoner**  | text, vision                 | text                     |
| **Generator** | text, vision, sound, action  | vision, sound, **action**|

The **Generator's `action` output is exactly a VLA policy**. The Cosmos 3
action surface has three modes:

| Mode              | `action_mode`       | Input                    | Output                          |
|-------------------|---------------------|--------------------------|---------------------------------|
| **Policy**        | `policy`            | image + instruction      | action chunk + rollout video    |
| Inverse dynamics  | `inverse_dynamics`  | video + instruction      | action chunk + video            |
| Forward dynamics  | `forward_dynamics`  | image + action chunk     | video                           |

For the robots `Policy` contract we care about **`policy` mode**:
`observation (image) + instruction → action chunk`. That is a 1:1 match with
`Policy.get_actions(observation_dict, instruction) -> list[dict]`.

Checkpoint: **`nvidia/Cosmos3-Nano-Policy-DROID`** (16B) — fits a single L40S.

---

## 1. The robots Policy contract (target interface)

From `strands_robots/policies/base.py`:

```python
class Policy(ABC):
    async def get_actions(self, observation_dict, instruction, **kwargs) -> list[dict]
    def get_actions_sync(...)              # provided wrapper
    def set_robot_state_keys(self, keys)   # abstract
    def reset(self, seed=None)             # optional override
    @property requires_images -> bool      # default True
    @property provider_name -> str         # abstract
```

`get_actions` returns a **list of per-timestep action dicts** (an action chunk),
e.g. `[{"joint_0": 0.1, "gripper": 0.0}, ...]`. Our Cosmos3 policy must:
1. Take the robot observation (camera frame(s) + state) + instruction.
2. Produce a Cosmos3 `policy` request (image + instruction + embodiment params).
3. Receive the predicted action chunk (10D DROID end-effector + gripper).
4. Map model action keys → robot actuator names (like `Gr00tPolicy`).

### Reference patterns to mirror
- **`Gr00tPolicy`** (`policies/groot/policy.py`): service-mode (ZMQ) vs local-mode,
  explicit `ObservationMapping` / `ActionMapping`, `_unpack_actions`, `reset()`
  forwarding to server. **This is our primary template** — Cosmos3 is also a
  service-backed VLA returning action chunks.
- **`MockPolicy`** (`policies/mock.py`): minimal shape.
- **Plugin registry**: `registry/policies.json` (declarative provider entry) +
  `factory.create_policy` (smart-string resolution + trust-remote-code gate).

---

## 2. Cosmos 3 action I/O — exact format (from `../cosmos` cookbook)

### Action representation (unified interface, from paper)
- Ego / end-effector pose = **9D pose delta**: 3D translation + 6D continuous rotation.
- Grasp state: **1D** open/close (grippers) or 15D hand.

| Embodiment        | Repr                          | Dim | Unit  | Chunk            |
|-------------------|-------------------------------|-----|-------|------------------|
| DROID             | EE pose 9D + gripper 1D       | 10D | meter | 16 frames @ 15FPS|
| UMI               | EE pose 9D + gripper 1D       | 10D | meter | 16 frames @ 20FPS|
| Autonomous vehicle| Ego pose 9D                   | 9D  | meter | 60 frames @ 10FPS|

Action JSON = list of per-step vectors, e.g. DROID/UMI = list of 10-floats:
```json
[[tx, ty, tz, r1..r6, grasp], ...]   # one row per timestep
```
(verified against `cookbooks/.../assets/actions/umi.json` = 10D rows,
`av_traj_forward.json` = 9D rows).

### DROID embodiment layout (from `droid_lerobot_example/meta/info.json`)
- cameras: `observation.image.exterior_image_1_left`, `exterior_image_2_left`,
  `wrist_image_left` (360×640×3, 15 fps)
- state: `observation.state.cartesian_position` (6), `joint_positions`, `gripper_position`
- action: `action.joint_position`, `action.gripper_position`

### Backends (pick service-first to match groot)
| Backend             | Entry                                            | Robots use            |
|---------------------|--------------------------------------------------|-----------------------|
| **vLLM-Omni** ✅     | `POST /v1/videos` async job, `action_mode=policy`| **service mode (primary)** |
| Cosmos Framework    | `torchrun -m cosmos_framework.scripts.inference` | local/offline (secondary)  |

**vLLM-Omni request (policy mode):**
- multipart `POST /v1/videos` (async job — policy returns an action chunk)
- `files={"input_reference": <start image>}`
- `--form-string "prompt=<instruction>"`
- `extra_params={"action_mode":"policy","domain_name":"droid",
   "raw_action_dim":10,"action_chunk_size":16, "guardrails":false}`
- diffusion defaults for action: `num_inference_steps=30`, `guidance_scale=1.0`,
  `flow_shift=10.0`
- Poll job → read predicted action chunk from completed result.

Server (Docker, all modalities incl. action):
```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v "$(pwd):/workspace" \
  -p 8000:8000 --ipc=host vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano-Policy-DROID --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / --port 8000 --init-timeout 1800
```

---

## 3. Proposed package layout

```
strands_robots/policies/cosmos3/
    __init__.py          # exports Cosmos3Policy + mappings
    policy.py            # Cosmos3Policy(Policy) — service + local modes
    client.py            # Cosmos3OmniClient — vLLM-Omni /v1/videos job client
    embodiments.py       # DROID / UMI / AV embodiment specs (dims, cams, action layout)
```
Plus registry wiring in `strands_robots/registry/policies.json`.

### `Cosmos3Policy` design (mirrors Gr00tPolicy)
```python
class Cosmos3Policy(Policy):
    def __init__(
        self,
        embodiment: str = "droid",          # droid | umi | av
        host: str = "localhost",
        port: int = 8000,                   # vLLM-Omni server
        model_path: str | None = None,      # local mode (Cosmos Framework) → torchrun
        domain_name: str | None = None,     # override embodiment domain
        raw_action_dim: int | None = None,  # default from embodiment spec
        action_chunk_size: int = 16,
        observation_mapping: dict | None = None,  # {robot_cam: "image.primary"}
        action_mapping: dict | None = None,       # {"action.ee_pose": "joint_position", ...}
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        flow_shift: float = 10.0,
        guardrails: bool = False,
        seed: int = 0,
        **kwargs,
    ): ...

    async def get_actions(self, observation_dict, instruction, **kwargs):
        # 1. select conditioning image from observation_dict via obs mapping
        # 2. build vLLM-Omni policy request (start image + instruction + extra_params)
        # 3. submit async /v1/videos job, poll to completion
        # 4. parse predicted action chunk (list[list[float]])
        # 5. map model action dims → robot actuator keys → list[dict] (chunk)

    def set_robot_state_keys(self, keys): ...   # store; used for fallback mapping
    def reset(self, seed=None): ...             # forward seed to server job / reseed local
    @property provider_name -> "cosmos3"
    @property requires_images -> True
```

### `embodiments.py` (data-driven, like groot data_config)
```python
@dataclass(frozen=True)
class Cosmos3Embodiment:
    name: str                # "droid"
    domain_name: str         # vLLM-Omni domain (e.g. "droid", "bridge_orig_lerobot")
    raw_action_dim: int      # 10 (DROID)
    action_chunk_size: int   # 16
    fps: int                 # 15
    camera_keys: list[str]   # robot-side camera names this embodiment expects
    action_layout: list[str] # ["tx","ty","tz","r1".."r6","grasp"] → for key mapping
EMBODIMENTS = {"droid": ..., "umi": ..., "av": ...}
```

---

## 4. Registry entry (`registry/policies.json`)

```json
"cosmos3": {
  "module": "strands_robots.policies.cosmos3",
  "class": "Cosmos3Policy",
  "description": "NVIDIA Cosmos 3 omnimodal VLA policy (DROID/UMI/AV) via vLLM-Omni",
  "requires": [],
  "config_keys": [
    "embodiment", "host", "port", "model_path", "domain_name",
    "raw_action_dim", "action_chunk_size", "num_inference_steps",
    "guidance_scale", "flow_shift", "guardrails", "seed"
  ],
  "defaults": {"host": "localhost", "port": 8000, "embodiment": "droid"},
  "shorthands": ["cosmos3", "cosmos", "c3"],
  "url_patterns": ["^cosmos3://"],
  "hf_orgs": ["nvidia"],
  "model_id_overrides": ["nvidia/cosmos3", "nvidia/cosmos3-nano-policy"]
}
```
Note: `nvidia` org already maps to groot via `hf_orgs`. To disambiguate, use
`model_id_overrides` prefix `nvidia/cosmos3...` (checked before `hf_orgs` in
`resolve_policy`) so `create_policy("nvidia/Cosmos3-Nano-Policy-DROID")` → cosmos3.
Also gate under `_HF_REMOTE_CODE_PROVIDERS` in `factory.py` if local mode loads
with `trust_remote_code=True`.

---

## 5. Phased execution (run + verify each gate)

### Phase 0 — Plan + branch (THIS COMMIT)
- [x] Fork `cagataycali/robots`, branch `feat/cosmos3-policy`.
- [x] This plan committed.
- **Gate:** plan pushed to fork for review.

### Phase 1 — Local Cosmos 3 policy smoke (plain Python, no robots)
Run the model end-to-end locally first to learn its exact I/O.
- [ ] Start vLLM-Omni server (Docker `vllm/vllm-omni:cosmos3`) serving
      `nvidia/Cosmos3-Nano-Policy-DROID` on L40S. `curl /v1/models` OK.
- [ ] `scratch/c3_policy_smoke.py`: take DROID example start frame
      (from `../cosmos/cookbooks/.../droid_lerobot_example`), POST a `policy`
      job, poll, dump the returned action chunk shape + values.
- **Gate:** we have a concrete, verified action-chunk JSON for DROID (shape
  `[16, 10]`), and know the exact request/response envelope.

### Phase 2 — Client + embodiments
- [ ] `cosmos3/client.py` `Cosmos3OmniClient.policy(image, instruction, ...)` →
      action chunk (port the verified Phase-1 request/poll logic).
- [ ] `cosmos3/embodiments.py` with DROID/UMI/AV specs.
- [ ] Unit tests with a mocked HTTP server (no GPU): request shape + parse.
- **Gate:** `pytest tests/policies/test_cosmos3_client.py` green (mocked).

### Phase 3 — Cosmos3Policy provider
- [ ] `cosmos3/policy.py` implementing `Policy` (service mode via client).
- [ ] Obs/action mapping (auto-infer + explicit, mirror groot helpers).
- [ ] `reset(seed)` forwards seed; `requires_images=True`.
- [ ] Register in `policies.json` + `policies/__init__.py` exports.
- [ ] `create_policy("cosmos3", embodiment="droid", port=8000)` constructs.
- **Gate:** unit test constructs policy + `get_actions` returns `list[dict]`
  with mapped keys against a mocked client.

### Phase 4 — End-to-end on hardware
- [ ] Real server up; `get_actions_sync` on the DROID example frame +
      "pick up the object" → valid action chunk.
- [ ] Optional: run inside a sim/eval loop (LIBERO/MuJoCo) if embodiment maps.
- **Gate:** real action chunk produced through the full robots Policy path.
  **→ ASK USER TO VERIFY before push.**

### Phase 5 — Local (Cosmos Framework) mode + docs
- [ ] Optional local mode: `torchrun -m cosmos_framework.scripts.inference`
      (offline, no server) behind `model_path=`.
- [ ] `examples/` snippet, README policy table + AGENTS.md learning log entry.
- [ ] Expand tests; CPU-only import gate in CI.
- **Gate:** docs build, example runs, `create_policy("cosmos3")` documented.

---

## 6. Risks & open questions (confirm while testing)
- **vLLM-Omni action mode availability**: action output is in follow-up PRs;
  the `vllm/vllm-omni:cosmos3` Docker image is required (PR branch covers only
  t2i/t2v/i2v). Confirm the image exposes `action_mode=policy` on `/v1/videos`.
- **Action chunk parsing**: where in the completed job result does the action
  array live? (Phase 1 answers this — dump full response.)
- **Coordinate conventions**: DROID needs `to-OpenCV` + multiview concat +
  normalization (per cookbook). Confirm whether the server returns raw or
  post-processed action; map to robot actuator space accordingly.
- **State conditioning**: Cosmos3 policy is image+instruction (no explicit joint
  state input in the request). Robot state goes unused for conditioning but is
  still needed for `set_robot_state_keys` / action key mapping.
- **DROID vs robot embodiment mismatch**: the policy emits DROID 10D EE-pose
  deltas; downstream robot must accept EE-delta control (or we add an IK/decoder
  shim). Document this constraint; default target = DROID-like arms.
- **Latency**: diffusion-based action gen is slow (not 500Hz). Suitable for
  chunked control, not high-rate servo. Action chunk amortizes this (16 steps).

## 7. Decisions locked
1. NIM excluded. vLLM-Omni Docker = primary action backend; Cosmos Framework = secondary local.
2. Default checkpoint: `nvidia/Cosmos3-Nano-Policy-DROID` (fits L40S).
3. Provider mirrors `Gr00tPolicy` (service + local, explicit mappings).
4. Plugin registry entry in `policies.json`; smart-string `cosmos3://` + nvidia/cosmos3 IDs.
5. Contract preserved: `Cosmos3Policy(Policy)` returns action chunks as `list[dict]`.
6. Run locally first (Phase 1 scratch script) → verify I/O → then implement.
