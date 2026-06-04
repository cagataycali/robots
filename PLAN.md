# PLAN: Qwen-VLA Integration into strands-robots

> Feature branch: `feat/qwen-vla`
> Target: integrate the **Qwen-VLA** unified vision-language-action model
> (Qwen Team, arXiv:2605.30280v2, June 2026) as a first-class policy provider,
> AND wire up the full **4-stage training recipe** (T2A → CPT → SFT → RL)
> so `strands-robots` can be used end-to-end to **collect data → post-tune
> Qwen-VLA → continue / redeploy**.

---

## 0. TL;DR

Qwen-VLA = Qwen3.5-4B VLM backbone + 1.15B-param DiT flow-matching action
expert, conditioned on an **embodiment-aware text prompt** (the sole
platform-specific interface). It treats manipulation, navigation, and
trajectory prediction as one shared action-and-trajectory prediction problem.

strands-robots already has everything we need to host it as a generalist policy
and to drive its training loop:

| Qwen-VLA needs | strands-robots already provides |
|---|---|
| Inference policy interface | `policies/base.Policy` + `policies.json` registry |
| Server OR local inference modes | mirror `groot/policy.py` (ZMQ service + local) |
| Embodiment prompt | `Gr00tDataConfig` + per-robot registry metadata |
| Action chunk (H=16) execution | `PolicyRunner.run` / `action_horizon` |
| Data collection (T2A/CPT/SFT corpora) | `DatasetRecorder` → LeRobotDataset |
| Eval / RL rollouts (success reward) | `PolicyRunner.evaluate` + `BenchmarkProtocol` |
| Multi-embodiment fleet | `registry/robots.json`, MuJoCo sim, mesh |

The integration is **two halves**:
1. **Inference provider** (`policies/qwen_vla/`) — deploy & run a Qwen-VLA
   checkpoint inside the existing Robot/Sim/Agent stack. (Ships first.)
2. **Training pipeline** (`strands_robots/training/qwen_vla/`) — the 4-stage
   recipe driven by data collected through this repo. (Ships incrementally.)

---

## 1. Architecture Mapping

### 1.1 Qwen-VLA model surface (from the paper)

- **Backbone:** Qwen3.5-4B (early-fusion multimodal, hybrid gated-linear +
  grouped-query softmax attention).
- **Action expert:** single-stream DiT (16 blocks, ~1.15B params),
  flow-matching objective, AdaLN timestep conditioning, multi-section RoPE.
- **I/O contract:**
  - Inputs: `o_t` (1+ camera frames), `x` (instruction), `e` (embodiment
    prompt), optional `z` (task id).
  - Output: `Y ∈ R^{H×K}` — H=16 horizon, K = fixed channel dim, leading
    `c` channels valid + zero-padded tail, per-channel binary mask `M`.
- **Embodiment prompt template** (verbatim from §2.3):
  ```
  The robot is {robot_tag} with {single arm / dual arms}[, waist][, and mobile
  base]. The control frequency is {FPS} Hz. Please predict the next {chunk_size}
  control actions to execute the following task: {ori_instruction}.
  ```
- **Action families:** manipulation (`Δx,Δy,Δz` + rotation + gripper +
  dexterous-hand), navigation (`Δx,Δy,Δθ` waypoints), egocentric (wrist SE(3)
  + 10 eigengrasp coeffs/hand → 32 dims).
- **Normalization:** per-dataset quantile (1st/99th pct → `[-1,1]`, eq. 5).
- **Camera view tokens:** `<|tag_start|> <image> <|tag_end|>` with
  `ego` / `cam_left_wrist` / `cam_right_wrist`.

### 1.2 Where it slots into this repo

```
observation (cameras + joints)
        │  strands_robots PolicyRunner / Robot tool
        ▼
QwenVlaPolicy.get_actions(obs, instruction)        ← NEW (policies/qwen_vla/)
        │  builds embodiment prompt e from QwenVlaDataConfig
        │  packs video/state into model-native batch
        ▼
[ LOCAL: in-proc Qwen-VLA ] OR [ SERVICE: ZMQ/HTTP server ]
        │  flow-matching DiT → action chunk Y[H×K]
        ▼
unpack → list[dict] (one per timestep) → robot actuators
```

Mirror the GR00T design exactly: explicit `observation_mapping` /
`action_mapping`, no positional guessing, validate against model modality
config, support both LOCAL and SERVICE modes, forward `reset(seed=)` to the
server for reproducible RL/eval (the #187 lesson).

---

## 2. Deliverables & Directory Layout

```
strands_robots/
├── policies/
│   └── qwen_vla/                      # NEW — inference provider
│       ├── __init__.py                # thin exports (QwenVlaPolicy)
│       ├── policy.py                  # QwenVlaPolicy (LOCAL + SERVICE)
│       ├── client.py                  # QwenVlaInferenceClient (ZMQ/HTTP)
│       ├── data_config.py             # QwenVlaDataConfig + embodiment prompt builder
│       ├── data_configs.json          # per-embodiment configs (so100, aloha, g1, widowx...)
│       ├── prompt.py                  # embodiment-aware prompt template (§2.3)
│       └── normalize.py               # per-dataset quantile norm (eq.5) + channel mask
├── training/                          # NEW — training pipeline (half 2)
│   └── qwen_vla/
│       ├── __init__.py
│       ├── config.py                  # dataclasses for all 4 stages
│       ├── data/
│       │   ├── mixture.py             # weighted multi-source sampler (Table 1)
│       │   ├── lerobot_adapter.py     # LeRobotDataset → Qwen-VLA tensors
│       │   ├── language_action.py     # text-only T2A corpus generator
│       │   └── embodiment_tags.py     # robot → embodiment prompt registry
│       ├── stage1_t2a.py              # text-to-action DiT pretraining
│       ├── stage2_cpt.py              # continued pretraining (vision grounding)
│       ├── stage3_sft.py              # multi-task supervised fine-tuning
│       ├── stage4_rl.py               # PPO + GAE on sim success reward
│       ├── flow_matching.py           # objective, timestep dists (Beta/Sig-Norm)
│       └── ppo/
│           ├── rollout.py             # client-server rollout (N=128 envs)
│           ├── logprob.py             # flow-matching logπ via SDE (Song'21)
│           └── value_head.py          # VLM-attached value head (stop-grad)
├── tools/
│   ├── qwen_vla_inference.py          # NEW @tool — deploy/run server (mirror gr00t_inference.py)
│   └── qwen_vla_train.py              # NEW @tool — launch training stage from agent
└── registry/
    └── policies.json                  # ADD "qwen_vla" provider entry

tests/
├── test_qwen_vla_policy.py            # unit: prompt build, mapping, unpack, mask
├── test_qwen_vla_data_config.py       # unit: config resolution + prompt template
├── test_qwen_vla_normalize.py         # unit: quantile norm round-trip + mask
└── test_qwen_vla_training_config.py   # unit: stage config validation
tests_integ/
├── test_qwen_vla_inference.py         # real checkpoint inference (GPU-gated)
├── test_qwen_vla_t2a.py               # tiny T2A smoke (few steps)
└── test_qwen_vla_sft_eval.py          # SFT→eval loop on LIBERO/sim (GPU-gated)

pyproject.toml                         # ADD [project.optional-dependencies] qwen-vla / qwen-vla-train
docs/                                  # ADD qwen_vla.md usage + training guide
```

---

## 3. Phase Breakdown (incremental, each a mergeable PR)

### Phase A — Inference Provider (LOCAL + SERVICE)  ← ship first
**Goal:** run an existing Qwen-VLA checkpoint through `Robot()` / `PolicyRunner`.

1. `policies/qwen_vla/prompt.py` — `build_embodiment_prompt(cfg, instruction)`
   implementing the §2.3 template exactly. Pure function, fully unit-tested.
2. `policies/qwen_vla/data_config.py` + `data_configs.json` — typed
   `QwenVlaDataConfig` (mirror `Gr00tDataConfig`): `video_keys`, `state_keys`,
   `action_keys`, `robot_tag`, `arm_config` (single/dual), `has_waist`,
   `has_mobile_base`, `fps`, `chunk_size` (H=16), `image_view_tags`,
   `quantile_stats_path`. Seed configs: `so100`, `aloha_bimanual`,
   `widowx`, `unitree_g1`, `franka_panda`, `libero_panda`.
3. `policies/qwen_vla/normalize.py` — per-dataset quantile normalize/unnormalize
   (eq. 5) + `build_channel_mask(c, K, H_task, H)`.
4. `policies/qwen_vla/client.py` — `QwenVlaInferenceClient` (ZMQ first; reuse
   `groot/client.py` msgpack pattern; optional HTTP later).
5. `policies/qwen_vla/policy.py` — `QwenVlaPolicy(Policy)`:
   - `__init__(model_path=... → LOCAL | host/port → SERVICE)`.
   - `observation_mapping` / `action_mapping` (explicit, validated, auto-infer
     fallback) — copy GR00T's parse/validate machinery.
   - `get_actions`: build prompt `e`, pack `{video, state, language}`, run
     flow-matching inference (few Euler steps), unpack `Y[H×K]` → per-step dicts
     using the channel mask + `action_mapping`.
   - `reset(seed=)`: LOCAL reseed + SERVICE forward (the #187 contract).
   - `requires_images = True`.
6. `registry/policies.json` — add `qwen_vla` provider (module/class, config_keys,
   shorthands `["qwen", "qwen-vla"]`, `hf_orgs: ["Qwen", "QwenLM"]`,
   `url_patterns` for `zmq://`). Gate `trust_remote_code` if the HF load path
   needs it (add to `_HF_REMOTE_CODE_PROVIDERS` in `factory.py`).
7. `tools/qwen_vla_inference.py` — `@tool` to start/stop a server + run inference,
   mirroring `gr00t_inference.py`, with a `validate_inputs()` allowlist
   (PR #90/#92 lesson: allowlist `data_config`, reject shell metacharacters,
   bind `127.0.0.1`).
8. `pyproject.toml` — `qwen-vla = ["torch>=2.4,<3.0", "transformers>=4.46,<5.0",
   "qwen-vla-pkg ...", "pyzmq", "msgpack"]` (exact deps pinned once the model
   release pkg name is known). Add to `all`.
9. Tests: unit (prompt, config, normalize, mapping, unpack) + 1 GPU-gated integ.

**Exit criteria:** `Robot("so100", policy="qwen_vla", model_path=...)` runs a
rollout in MuJoCo sim and on real hardware; `PolicyRunner.evaluate` reports a
success rate on LIBERO.

### Phase B — Data Collection for Training
**Goal:** turn this repo into the data engine for the 4-stage recipe.

1. `training/qwen_vla/data/embodiment_tags.py` — map every robot in
   `registry/robots.json` to its embodiment prompt fields. Single source of truth
   shared by inference (`data_config.py`) and training.
2. Extend `DatasetRecorder` usage docs/helpers so teleop + sim rollouts emit
   LeRobotDatasets already tagged with: embodiment prompt, fps, chunk_size,
   camera view tags, per-dataset quantile stats. (DatasetRecorder itself already
   does the heavy lifting — we add a thin `qwen_vla` schema preset.)
3. `training/qwen_vla/data/lerobot_adapter.py` — LeRobotDataset → Qwen-VLA
   `(video, state, language, Y[H×K], mask)` tensors with the unified
   zero-padding channel layout (§2.4) + quantile norm.
4. `training/qwen_vla/data/language_action.py` — generate the **text-only T2A
   corpus** (§3.2.3 language-action): 6 task-template families × robot configs,
   procedural instructions, motion-planned EEF goals (reuse `cuRobo`/IK if
   available, else MuJoCo IK). No rendering — fast, scalable.
5. `training/qwen_vla/data/mixture.py` — weighted sampler reproducing Table 1
   proportions (manip 74.2% / nav 7.5% / egocentric 6% / sim 3.7% / VL 8.5%),
   configurable so users can up-weight their own collected data.

**Exit criteria:** `collect → DatasetRecorder → adapter` produces a training
batch that `stage*.py` can consume; a smoke test loads N frames and asserts
tensor shapes + mask correctness.

### Phase C — Stage 1: Text-to-Action (T2A) Pretraining
**Goal:** train the DiT action prior from language alone (vision frozen/absent).

- `flow_matching.py`: conditional flow-matching loss (eqs. 1-2), two-level
  per-channel averaging with mask, **Sigmoid-Normal** timestep dist for T2A
  (paper's best, §5.2.1), Euler integration at inference.
- `stage1_t2a.py`: freeze VLM, train DiT only, **no images**, full-sequence
  prediction (not chunk — +4.9pp per ablation), ~2k steps default
  (paper's sweet spot), ~20% synthetic + 80% real mix.
- Config knobs surfaced in `config.py` with the paper's defaults as documented
  constants (cite §5.2.1 in docstrings).

**Exit criteria:** tiny T2A run converges on a toy corpus; checkpoint loads into
Phase D as decoder warm-start.

### Phase D — Stage 2: Continued Pretraining (CPT)
**Goal:** unfreeze both modules, ground the action prior in vision.

- `stage2_cpt.py`: joint VLM+DiT training on the heterogeneous mixture
  (Phase B), **Beta** timestep dist (§5.2.1), VL co-training loss (eq. 3, weight
  0.1 VL / 1.0 action) to prevent catastrophic forgetting.
- Zero-padding projection for heterogeneous action dims (§5.2.2 — lightest,
  no per-embodiment heads).
- Produces **Qwen-VLA-Base**.

### Phase E — Stage 3: Supervised Fine-Tuning (SFT)
**Goal:** specialize Base on curated target-task demos.

- `stage3_sft.py`: two tracks — (a) multi-task (VQA + grounding + manip + nav),
  (b) real-robot teleop track from this repo's `DatasetRecorder` output.
- Loss weights 0.1 VL / 1.0 action; H=16 manip / 8 nav waypoints.

### Phase F — Stage 4: Reinforcement Learning (PPO + GAE)
**Goal:** optimize closed-loop task success in sim.

- `ppo/rollout.py`: decoupled client-server rollout using
  `PolicyRunner.evaluate` + `BenchmarkProtocol` as the env; N parallel envs;
  sparse binary success reward `R∈{0,1}`; embodiment prompt identical to SFT.
- `ppo/logprob.py`: flow-matching logπ via probability-flow ODE→SDE conversion
  (Song et al. 2021), single random denoising step per rollout (§4.2).
- `ppo/value_head.py`: lightweight value head on VLM hidden states with
  **stop-gradient** (paper: value LR ≈ 20× actor LR).
- Action-chunk-level credit assignment (one scalar reward + advantage per H=16
  chunk). GAE γ=0.99, λ=0.95, ε=0.2, 4 epochs/batch.
- Produces **Qwen-VLA-Instruct**.

**Exit criteria:** `+RL` checkpoint shows non-negative transfer on held-out sim
benchmarks (reproduce the Table 11 trend on a small scale).

### Phase G — Continuous Loop / Redeploy
- `tools/qwen_vla_train.py` agent tool: trigger any stage, resume from
  checkpoint, push trained model to HF, hot-swap into a running
  `QwenVlaPolicy` SERVICE.
- Wire the mesh (`strands_robots/mesh/`) so a fleet of robots collects data →
  central trainer post-tunes → redeploys updated checkpoint. Closes the
  collect → tune → continue loop the user asked for.

---

## 4. Key Design Decisions (and why)

1. **Mirror GR00T, don't reinvent.** GR00T's `policy.py` already solved the hard
   problems: explicit obs/action mapping + validation, LOCAL vs SERVICE,
   `reset(seed=)` server forwarding (#187), wire-payload diagnostics, image
   rotation quirks. Qwen-VLA's nested I/O is analogous → reuse the patterns.
2. **Embodiment prompt is THE interface.** Per the paper, the only
   platform-specific input is the text prompt. So `QwenVlaDataConfig` carries
   prompt fields, NOT per-robot model heads. Deploy to a new robot = new prompt,
   zero architecture change (§4 out-of-domain generalization).
3. **Zero-padding action layout** (§5.2.2) — single DiT param set, mask excludes
   padding from the loss/gradient. Cheapest, paper-default.
4. **Training is opt-in + heavy.** Gate behind `qwen-vla-train` extra (torch,
   transformers, accelerate/deepspeed, flow-matching deps). Inference stays
   lighter under `qwen-vla`.
5. **Sim is the RL env.** `PolicyRunner.evaluate` + `BenchmarkProtocol` already
   yields seeded, reproducible, success-scored rollouts → reuse as the PPO
   environment instead of a bespoke harness.
6. **DatasetRecorder is the data spine.** All collected demos (teleop + sim +
   replay) already become LeRobotDatasets; the adapter is the only new bridge.

---

## 5. Conventions Compliance (AGENTS.md)

- Python 3.12+, dependency bounds (`>=1.0`→cap major, `<1.0`→cap minor).
- Thin `__init__.py` (exports only).
- **Raise on fatal**, no silent zero-action defaults, no `except Exception`
  for non-recovery paths.
- `require_optional()` for torch/transformers/qwen pkg.
- `validate_inputs()` allowlist on every `@tool` param flowing into
  subprocess / paths (PR #92).
- No emojis in tool result dicts / logs / errors (plain ASCII).
- Bind servers to `127.0.0.1` by default.
- Integration tests per provider with real inference (GPU-gated via
  `pytest.importorskip`).
- Pin every reviewed fix with a regression test; no host paths in tests.
- Register every new `STRANDS_*` env var in README Configuration
  (e.g. `STRANDS_QWEN_VLA_WIRE_LOG`, training knobs).

---

## 6. Open Questions / Risks

1. **Checkpoint availability:** is a Qwen-VLA checkpoint + inference package
   public yet? If not, Phase A LOCAL mode is blocked on the upstream release;
   SERVICE mode can stub against a local server contract in the meantime.
2. **Package name / load path:** confirm the `pip` package + HF org
   (`Qwen` vs `QwenLM`) and whether loading needs `trust_remote_code`.
3. **Compute for training:** Phases C-F need multi-GPU (DiT 1.15B + 4B backbone).
   Decide DeepSpeed/FSDP strategy; expose via `config.py`.
4. **RLinf vs in-repo PPO:** the paper uses RLinf. Decide whether to vendor a
   minimal PPO (our `ppo/`) or integrate RLinf as an optional backend.
5. **cuRobo dependency** for language-action data gen — heavy; provide a MuJoCo
   IK fallback so the T2A corpus generator works without it.
6. **Action-space heterogeneity** across this repo's robots — validate the
   zero-padding `K` is large enough for the widest embodiment (dual-arm + dex
   hand).

---

## 7. Milestone Order (PR-by-PR)

1. **PR1 (Phase A.1-A.3):** prompt + data_config + normalize + unit tests.
2. **PR2 (Phase A.4-A.9):** client + policy + registry + tool + integ test.
   → *Qwen-VLA inference usable end-to-end.*
3. **PR3 (Phase B):** embodiment tags + adapter + mixture + smoke tests.
4. **PR4 (Phase C):** flow-matching + T2A stage + tiny-run test.
5. **PR5 (Phase D):** CPT stage.
6. **PR6 (Phase E):** SFT stage.
7. **PR7 (Phase F):** PPO/GAE RL stage.
8. **PR8 (Phase G):** train tool + mesh redeploy loop + docs.

Each PR: `hatch run format && hatch run lint && hatch run test` clean,
tracked on the project board (PVT_kwDOD151Fs4BSRJP) with Status + Priority.

---

## 8. References

- Qwen-VLA paper: arXiv:2605.30280v2 (§2 model, §3 pretraining, §4 post-train,
  §5 experiments + ablations).
- Existing patterns to mirror: `policies/groot/policy.py`,
  `policies/lerobot_local/policy.py`, `dataset_recorder.py`,
  `simulation/policy_runner.py`, `simulation/benchmark.py`.
