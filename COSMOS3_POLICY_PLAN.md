# Cosmos 3 Policy Provider — Implementation Plan (strands-robots)

> Branch: `feat/cosmos3-policy` (fork: `cagataycali/robots`)
> Goal: Add **NVIDIA Cosmos 3** as a first-class VLA **policy provider** in
> `strands_robots.policies`, so `create_policy("cosmos3", ...)` returns a
> `Policy` that maps robot observations → action chunks using
> **`nvidia/Cosmos3-Nano-Policy-DROID`** (16B vision-language robot policy).
>
> Constraint: **No NIM. Local compute only** (1× L40S 46 GB, driver 580 → CUDA 13.0).
> Backend: **Cosmos Framework** (pure Python, `git@github.com:NVIDIA/cosmos-framework.git`).
> Run the model locally first, verify the I/O, then wire it into the contract.

---

## 0. Why Cosmos 3 = a robots Policy

Cosmos 3 = omnimodal world model (MoT). Its **Generator `action` output IS a VLA
policy**. The `policy` action mode = `image + instruction → action chunk` — a 1:1
match with `Policy.get_actions(observation_dict, instruction) -> list[dict]`.

Checkpoint: **`nvidia/Cosmos3-Nano-Policy-DROID`** (16B) — fits a single L40S.

---

## 1. ⭐ KEY DISCOVERY — Cosmos Framework ships a ready-made policy SERVER

`cosmos-framework` is **pure Python** (no NIM, no Docker required, no vLLM-Omni
PR-branch limits). It ships a **DROID policy WebSocket server** that we wrap
exactly like groot's ZMQ service mode:

`cosmos_framework/scripts/action_policy_server_robolab.py`

```bash
PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
    --port 8000
# → ws://<ip>:8000/   (OpenPI WebsocketPolicyServer, msgpack+NumPy)
# → http://<ip>:8000/healthz
```

It uses **OpenPI's `WebsocketPolicyServer`** (the openpi msgpack+NumPy protocol):
- on connect → sends empty metadata dict
- each client message = an **observation dict**
- each response = `{"action": np.ndarray[T,D], "video"?: np.ndarray}`

→ **Our `Cosmos3Policy` service mode = an OpenPI websocket _client_.** This is
the same shape as `Gr00tPolicy` service mode (server holds the GPU model;
client sends obs, gets action chunk). OpenPI websocket clients are standard.

### Verified server I/O contract (from robolab server source)

**Observation dict (client → server):** keys use `/` separators
| Key | Shape / type | Notes |
|-----|--------------|-------|
| `prompt` | `str` | the instruction |
| `observation/image` | `[H,W,3] uint8` | single view, OR use 3-cam below |
| `observation/wrist_image_left` | `[H,W,3] uint8` | RoBoArena multiview (auto-composed) |
| `observation/exterior_image_1_left` | `[H,W,3] uint8` | composed: wrist on top, two exts below |
| `observation/exterior_image_2_left` | `[H,W,3] uint8` | |
| `observation/joint_position` | `[T,7]` float | `action_space="joint_pos"` (default) |
| `observation/gripper_position` | `[T,1]`/`[T]`/scalar | (server applies `1 - g`) |
| `observation/eef_pos` | `[T,3]` float | `action_space="midtrain"` |
| `observation/eef_quat` | `[T,4]` float (xyzw) | `action_space="midtrain"` |

**Action dict (server → client):**
| Key | Shape | Meaning |
|-----|-------|---------|
| `action` | `[T2, D]` | predicted chunk (T2 = chunk_size − history_length) |
| `video`  | `[T,H,W,3] uint8` | only if `--decode-video` |

**DROID defaults (released RoboLab policy):**
- `action_chunk_size=32`, `action_dim=8` (joint_pos) or `10` (midtrain)
- image `540×640`, `conditioning_fps=15`, `num_steps=4`, `guidance=3.0`, `shift=5.0`
- `joint_pos` action = `[7 joint deltas + 1 gripper]`
- `midtrain` action = `[3 pos + 4 quat(xyzw) + 1 gripper]` (abs pose, decoded server-side)

### Domains (`EMBODIMENT_TO_RAW_ACTION_DIM`, verified)
```
av=9, camera_pose=9, hand_pose=57, pusht=2, umi=10, bridge_orig_lerobot=10,
droid_lerobot=10, robomind-franka=10, robomind-franka-dual=20, robomind-ur=10,
agibotworld=29, fractal=10
```
DROID server default `domain_name="droid_lerobot"`.

### Offline path (no server) — also pure Python
`python -m cosmos_framework.scripts.inference -i <policy.json> -o <out> \
   --checkpoint-path Cosmos3-Nano` with `model_mode="policy"`. Outputs
`sample_outputs.json` (action) + `vision.mp4`. Input spec fields:
`domain_name, vision_path, prompt, action_chunk_size, fps, image_size, view_point`.
Used for batch / reproducibility; service mode is preferred for the live Policy.

---

## 2. The robots Policy contract (target)

From `strands_robots/policies/base.py`:
```python
class Policy(ABC):
    async def get_actions(self, observation_dict, instruction, **kwargs) -> list[dict]
    def get_actions_sync(...)             # provided
    def set_robot_state_keys(self, keys)  # abstract
    def reset(self, seed=None)            # optional override
    @property requires_images -> bool     # default True
    @property provider_name -> str        # abstract
```
`get_actions` returns a **list of per-timestep action dicts** (the chunk).

**Primary template: `Gr00tPolicy`** (`policies/groot/`): service-vs-local mode,
explicit `ObservationMapping`/`ActionMapping`, `_unpack_actions`, `reset()`
forwarding to server, msgpack client. We mirror this 1:1.

---

## 3. Package layout

```
strands_robots/policies/cosmos3/
    __init__.py        # exports Cosmos3Policy + mappings + embodiments
    policy.py          # Cosmos3Policy(Policy): service (websocket) + offline (subprocess) modes
    client.py          # Cosmos3WebsocketClient — OpenPI msgpack+NumPy client (thin)
    embodiments.py     # EMBODIMENTS: droid/umi/av/... (domain_name, raw_action_dim, cams, action_layout)
```
Plus registry wiring in `strands_robots/registry/policies.json`.

### `Cosmos3Policy` design (mirrors Gr00tPolicy)
```python
class Cosmos3Policy(Policy):
    def __init__(
        self,
        embodiment: str = "droid",            # droid | umi | av | bridge | ...
        host: str = "localhost", port: int = 8000,   # SERVICE mode (robolab ws server)
        checkpoint_path: str | None = None,   # OFFLINE mode (subprocess torchrun) → triggers local
        action_space: str = "joint_pos",      # joint_pos | midtrain (DROID)
        domain_name: str | None = None,       # default from embodiment
        action_chunk_size: int | None = None, # default 32 (DROID)
        observation_mapping: dict | None = None,  # {robot_cam: "observation/image"|"observation/wrist_image_left"...}
        action_mapping: dict | None = None,        # {model_action_index/name: robot_actuator}
        guidance: float = 3.0, num_steps: int = 4, shift: float = 5.0,
        conditioning_fps: float = 15.0, image_size: tuple[int,int] = (540, 640),
        seed: int = 0, decode_video: bool = False,
        **kwargs,
    ): ...

    async def get_actions(self, observation_dict, instruction, **kwargs):
        # 1. map robot obs → openpi obs dict (image(s)+state via observation_mapping)
        # 2. attach prompt=instruction
        # 3. client.infer(obs) → {"action": ndarray[T,D], "video"?: ...}
        # 4. split [T,D] rows → per-step robot actuator dicts via action_mapping
        #    (action_layout from embodiment names the D columns)
        # returns list[dict]  (the action chunk)

    def set_robot_state_keys(self, keys): ...   # store for fallback action key naming
    def reset(self, seed=None): ...             # service: send reset/seed hint; offline: reseed RNG
    @property provider_name -> "cosmos3"
    @property requires_images -> True
```

### `embodiments.py`
```python
@dataclass(frozen=True)
class Cosmos3Embodiment:
    name: str                 # "droid"
    domain_name: str          # "droid_lerobot"
    raw_action_dim: int       # 10 (midtrain) — model raw dim
    action_chunk_size: int    # 32
    fps: int                  # 15
    camera_keys: list[str]    # ["observation/wrist_image_left", "observation/exterior_image_1_left", ...]
    action_layout: dict[str, list[str]]   # {"joint_pos": ["j0".."j6","gripper"],
                                          #  "midtrain": ["x","y","z","qx","qy","qz","qw","gripper"]}
EMBODIMENTS = {"droid": ..., "umi": ..., "av": ..., "bridge": ...}
```

---

## 4. Registry entry (`registry/policies.json`)
```json
"cosmos3": {
  "module": "strands_robots.policies.cosmos3",
  "class": "Cosmos3Policy",
  "description": "NVIDIA Cosmos 3 omnimodal VLA policy (DROID/UMI/AV) via Cosmos Framework",
  "requires": [],
  "config_keys": [
    "embodiment","host","port","checkpoint_path","action_space","domain_name",
    "action_chunk_size","guidance","num_steps","shift","conditioning_fps","seed","decode_video"
  ],
  "defaults": {"host": "localhost", "port": 8000, "embodiment": "droid", "action_space": "joint_pos"},
  "shorthands": ["cosmos3","cosmos","c3"],
  "url_patterns": ["^cosmos3://"],
  "hf_orgs": ["nvidia"],
  "model_id_overrides": ["nvidia/cosmos3","nvidia/cosmos3-nano-policy"]
}
```
`model_id_overrides` (checked before `hf_orgs`) disambiguates from groot so
`create_policy("nvidia/Cosmos3-Nano-Policy-DROID")` → cosmos3. Offline mode loads
weights → add `"cosmos3"` to `_HF_REMOTE_CODE_PROVIDERS` in `factory.py`.

---

## 5. Phased execution (run + verify each gate)

### Phase 0 — Plan + branch ✅
- [x] Fork `cagataycali/robots`, branch `feat/cosmos3-policy`, plan committed.
- [x] Verified Cosmos Framework policy server I/O (this revision).

### Phase 1 — Stand up the framework + smoke the server (plain Python)
- [ ] `cd ../cosmos-framework && uv sync --all-extras --group=cu130-train --group=policy-server`
      (policy-server group pulls OpenPI WebsocketPolicyServer).
- [ ] `HF_TOKEN` set; `hf auth` ok; ensure GPU free (free the strands-cosmos reasoner).
- [ ] Launch robolab server with `nvidia/Cosmos3-Nano-Policy-DROID` on :8000; `curl /healthz`.
- [ ] `scratch/c3_policy_client_smoke.py`: build an openpi obs dict from the
      DROID example frame (`../cosmos/cookbooks/.../droid_lerobot_example`),
      send via openpi websocket client, dump returned `action` shape+values.
- **Gate:** concrete verified action chunk (e.g. `[32-h, 8]`); exact obs/action envelope confirmed.

### Phase 2 — Client + embodiments (+ mocked tests)
- [ ] `cosmos3/client.py` `Cosmos3WebsocketClient.infer(obs) -> dict` (port Phase-1 logic).
- [ ] `cosmos3/embodiments.py` DROID/UMI/AV/bridge specs.
- [ ] Unit tests with a fake websocket server (no GPU): obs encode + action decode.
- **Gate:** `pytest tests/policies/test_cosmos3_client.py` green (mocked).

### Phase 3 — Cosmos3Policy provider
- [ ] `cosmos3/policy.py` implementing `Policy` (service mode via client).
- [ ] Obs/action mapping (auto-infer + explicit, mirror groot helpers).
- [ ] `reset(seed)`, `requires_images=True`, register in `policies.json` + `__init__`.
- [ ] `create_policy("cosmos3", embodiment="droid", port=8000)` constructs.
- **Gate:** unit test: construct + `get_actions` → `list[dict]` w/ mapped keys (mocked client).

### Phase 4 — End-to-end on hardware  → **ASK USER TO VERIFY**
- [ ] Real robolab server up; `get_actions_sync(droid_frame, "pick up the object")` → action chunk.
- [ ] Optional: drive a sim/eval loop if embodiment maps.
- **Gate:** real chunk through the full robots Policy path. **Pause for user sign-off.**

### Phase 5 — Offline mode + docs + CI
- [ ] Offline mode: subprocess `cosmos_framework.scripts.inference model_mode=policy`
      behind `checkpoint_path=` (no server, batch/repro).
- [ ] `examples/` snippet, README policy table, AGENTS.md learning log.
- [ ] CPU-only import test in CI.
- **Gate:** docs/examples run; `create_policy("cosmos3")` documented.

---

## 6. Risks & open questions
- **OpenPI dependency**: service mode client needs `openpi-client` (msgpack+numpy
  websocket). Add as optional extra `cosmos3-service` (mirror groot's `groot-service`).
  Confirm exact client import path (`openpi_client.websocket_client_policy`).
- **Action semantics**: `joint_pos` = 7 joint deltas + gripper; `midtrain` = abs
  EE pose (pos+quat) + gripper. Pick what the target robot consumes; document that
  Cosmos3-DROID emits a Franka/DROID action space (IK/decoder shim may be needed
  for non-DROID arms).
- **State requirement**: server needs `observation/joint_position`(7)+`gripper`
  (joint_pos) or `eef_pos`+`eef_quat` (midtrain) in the obs — so `requires_images`
  stays True AND robot state must be provided. Map via `observation_mapping`.
- **Latency**: diffusion policy (~4 steps) is chunked, not 500Hz servo. Chunk of
  ~31 actions amortizes one inference. Fine for chunked control.
- **GPU memory**: 16B Nano-Policy + must free current 22 GB reasoner first.
- **action_chunk_size mismatch**: server trims `history_length` rows; returned T2
  = chunk − history. Use returned shape, don't assume.

## 7. Decisions locked
1. NIM excluded. **Cosmos Framework (pure Python)** = backend.
2. Default: `nvidia/Cosmos3-Nano-Policy-DROID`, `domain_name=droid_lerobot`.
3. **Service mode primary** = OpenPI websocket client → robolab policy server
   (mirrors Gr00tPolicy ZMQ). Offline `inference` subprocess = secondary.
4. Provider mirrors `Gr00tPolicy` (explicit obs/action mappings).
5. Registry entry + `cosmos3://` + `nvidia/Cosmos3...` ID resolution + trust-remote gate.
6. Contract preserved: `Cosmos3Policy(Policy)` returns action chunks as `list[dict]`.
7. Run locally first (Phase 1 server + client smoke) → verify I/O → implement.

## Phase 1 status (2026-06-04) — ✅ VERIFIED

Server + client smoke PASSED. `nvidia/Cosmos3-Nano-Policy-DROID` via the
framework robolab WebSocket server returns a `(32, 8)` action chunk
(32 steps x [7 joints + gripper]) from image+state+instruction. Warm latency
~3.1s/chunk. See `scratch/PHASE1_RESULTS.md` + `scratch/c3_policy_client_smoke.py`.
Gate cleared -> proceed to Phase 2 (client + embodiments + mocked tests).

## Phase 2+3 status (2026-06-04) — ✅ COMPLETE

Built the full Cosmos3 policy provider:
- `strands_robots/policies/cosmos3/{__init__,policy,client,embodiments}.py`
- registry entry in `policies.json` + export in `policies/__init__.py`
- `[cosmos3-service]` extra in pyproject (openpi-client)
- `tests/policies/cosmos3/` — 25 unit tests, all green (mocked, no GPU)
- live OpenPI client<->server roundtrip through Cosmos3Policy verified
- `PR.md` with color-coded mermaid diagrams, embodiment table, usage +
  MuJoCo episode-recording walkthrough.

`create_policy("cosmos3", embodiment="droid", port=8000)` is ready to use.
Gate cleared -> Phase 4 (full hardware rollout in sim) + Phase 5 (docs/examples).

## Phase 4 status (2026-06-04) — ✅ COMPLETE (live + recorded + pushed)

Real Cosmos3-Nano-Policy-DROID drove a Franka/Panda in MuJoCo; 3 episodes
recorded to LeRobotDataset (4 cams, 144 frames, 15fps) and pushed to HF:
https://huggingface.co/datasets/cagataydev/cosmos3-droid-mujoco
Videos embedded in PR.md (docs/media/cosmos3/). Scripts:
scratch/mujoco_record_episode.py + scratch/mujoco_record_multi.py.
Fixed Cosmos3Policy gripper mapping for finger_joint keys (+test). 26 tests green.
