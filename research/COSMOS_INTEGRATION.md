# Cosmos × strands-robots — Integration Research

**Author**: cagatay + DevDuck
**Date**: 2026-05-16
**Status**: Research / Design — pre-RFC
**Source repos analyzed**:
- `git@github.com:cagataycali/strands-cosmos.git` (your existing toolkit)
- `git@github.com:nvidia-cosmos/cosmos-predict2.5.git`
- `git@github.com:nvidia-cosmos/cosmos-transfer2.5.git`
- `git@github.com:strands-labs/robots.git` (this repo, current PR=#101 zenoh)

---

## TL;DR

There are **three orthogonal capabilities** that Cosmos brings, and each maps to
a different layer of strands-robots. Don't fold them all into one extension —
they need different abstractions and different optional-deps groups.

| Cosmos capability | strands-robots layer | New abstraction |
|---|---|---|
| **Cosmos-Reason2 VLM** (text+vision reasoning) | High-level agent / scene reasoning | NEW `Reasoner` interface (sibling of `Policy`) |
| **Cosmos-Predict2.5 robot/policy** (action-conditioned VLA) | `policies/` | NEW `CosmosPolicy` provider (peer of `Gr00tPolicy`, `LerobotLocalPolicy`) |
| **Cosmos-Predict2.5 base / Transfer2.5** (world-model video gen) | NEW `world_model/` subpackage | NEW `WorldModel` interface — sibling of `SimEngine` |

The right move is **NOT** to vendor any of the three Cosmos repos. It is to:

1. Take `strands-cosmos` as a **runtime dependency** (`pip install strands-cosmos`) gated behind `extras = [cosmos]`
2. Add **three thin adapters** in strands-robots: `policies/cosmos_predict/`, `reasoners/cosmos_reason/`, `world_model/cosmos/`
3. Keep upstream Cosmos repos **untouched** (mirrors strands-cosmos's own `UPSTREAM UNTOUCHED` rule)

This preserves three properties strands-robots already has:
- Lazy heavy imports (torch / cosmos / transformers stay out of `import strands_robots`)
- Backend-agnostic core (Cosmos becomes one provider among many)
- Optional-extras pattern (`pip install strands-robots[cosmos]`)

---

## 1. What Cosmos actually offers (mapped to robotics use-cases)

### 1.1 Cosmos-Reason2 — Physical-AI VLM

- **Models**: `nvidia/Cosmos-Reason2-2B` (24GB, edge — Jetson Thor/Orin), `Cosmos-Reason2-8B` (32GB, cloud)
- **Inputs**: video + image + text
- **Outputs**: text with optional `<think>...</think>` chain-of-thought
- **Already shipped in**: `strands-cosmos` as `CosmosVisionModel` (a Strands `Model`)
- **What it gives strands-robots**:
  - Scene understanding ("what is on the table?")
  - Affordance reasoning ("can the gripper grasp this?")
  - Failure analysis from rollout videos
  - Task decomposition from natural-language goals

**Today in strands-robots**: nothing comparable exists. The `Policy` interface
is low-level (observation → action chunk). There is no abstraction for
*high-level reasoning* over a scene.

### 1.2 Cosmos-Predict2.5 — World Foundation Model with Action Conditioning

Three variants matter for robotics:

| Variant | Input | Output | Use case |
|---|---|---|---|
| `Cosmos-Predict2.5-2B/robot/policy` | action chunks + image | action chunks | **VLA policy** (Libero, RoboCasa post-trained) — drop-in for ACT/Pi0 |
| `Cosmos-Predict2.5-2B/robot/action-cond` | action chunks + initial frame | predicted future video | World model: "if I do these actions, what will happen?" |
| `Cosmos-Predict2.5-2B/robot/multiview-agibot` | text + image | predicted multi-camera video | Multi-cam world model |
| `Cosmos-Predict2.5-2B/base` (+ distilled) | text + image/video | future video | Generic world generation |

**Critical insight**: the `robot/policy` variant is a **Policy in our taxonomy**
— it consumes (state, vision, language) and produces actions. The `robot/action-cond`
and `base` variants are **WorldModels** — they predict future observations.

This is why I argue for **two separate adapters**, not one.

### 1.3 Cosmos-Transfer2.5 — Multi-ControlNet Video-to-Video

- **Inputs**: depth, edge, segmentation, blur (any combo) + text prompt
- **Output**: photorealistic video matching the controls
- **Use cases for robotics**:
  - **Sim2Real augmentation**: render low-fi MuJoCo rollouts → photoreal videos
  - **Real2Real augmentation**: take dashcam → re-render in snow/night/fog
  - **Domain randomization at video level** (vs. our current MuJoCo-level randomization)
- **Distilled edge model** (Feb 2026 release) — fits on Jetson

**This is huge for the LeRobot dataset pipeline**. Today our `dataset_recorder.py`
records whatever MuJoCo renders. With Transfer2.5 we can **post-process** the
recorded videos into N photorealistic variants per episode → free dataset
multiplication.

---

## 2. How `strands-cosmos` already solves part of this

Your `strands-cosmos` package (v0.2.0, on PyPI) ships:

- `CosmosVisionModel` — Strands `Model` provider (text+image+video reasoning)
- 21 `@tool`-decorated functions covering: inference, predict_generate,
  transfer_generate, model lifecycle (download/quantize/onnx/trt), training
  (post_train/distill), data curation (Xenna), evaluation, RTP I/O, NATS pub.
- `justfile` as the source of truth for pipelines
- Verified on Jetson AGX Thor with the `fix_cublas.py` shim

**This means the heavy lifting is already done.** strands-robots should **not**
rebuild `cosmos_predict_generate` etc. — those are general-purpose Cosmos tools.
What strands-robots needs is the **robot-specific glue**: turning Cosmos
outputs into things our `Policy` / `Simulation` / `dataset_recorder` understand.

---

## 3. Proposed integration architecture

```
strands_robots/
├── policies/
│   ├── base.py                  (existing — Policy ABC)
│   ├── factory.py               (existing — create_policy)
│   ├── groot/                   (existing)
│   ├── lerobot_local/           (existing)
│   ├── mock.py                  (existing)
│   └── cosmos_predict/          ★ NEW
│       ├── __init__.py
│       ├── policy.py            CosmosPolicy(Policy) — wraps Cosmos-Predict2.5/robot/policy
│       ├── client.py            HF inference client (mirrors groot/client.py shape)
│       └── data_config.py       embodiment configs (Libero, RoboCasa, custom)
│
├── reasoners/                   ★ NEW SUBPACKAGE
│   ├── __init__.py
│   ├── base.py                  Reasoner ABC (high-level scene reasoning)
│   ├── factory.py               create_reasoner()
│   └── cosmos_reason/
│       ├── __init__.py
│       └── reasoner.py          CosmosReasoner(Reasoner) — thin wrapper over strands-cosmos.CosmosVisionModel
│
├── world_model/                 ★ NEW SUBPACKAGE
│   ├── __init__.py
│   ├── base.py                  WorldModel ABC (predict future obs)
│   ├── factory.py               create_world_model()
│   └── cosmos/
│       ├── __init__.py
│       ├── predict.py           CosmosPredictWorldModel — wraps cosmos_predict_generate
│       └── transfer.py          CosmosTransferAugmentor — sim2real video augmentation
│
├── tools/
│   ├── (existing tools)
│   └── cosmos_*.py              ★ NEW thin Strands @tool wrappers (only if useful at agent layer)
│
├── augmentation/                ★ NEW (optional, for dataset workflows)
│   ├── __init__.py
│   └── cosmos_transfer.py       offline LeRobotDataset video augmentation
│
└── registry/
    └── policies.json            ADD: "cosmos_predict" provider entry
```

### 3.1 Why three new abstractions, not one

`Policy`, `Reasoner`, `WorldModel` are **fundamentally different contracts**:

| Abstraction | Input | Output | Latency | Used by |
|---|---|---|---|---|
| `Policy` | obs (state+cam) + instruction | action chunk | ~10–100 Hz | control loop |
| `Reasoner` | obs/video + question | text (CoT) | ~1–10 Hz | agent planning |
| `WorldModel` | obs + action plan | future video/obs | offline / ~1 Hz | dream-rollout, eval, data aug |

Forcing them into one interface would either bloat `Policy` or hide capability.
Three clean ABCs is what the `simulation/` redesign already taught us: **separate
interfaces let backends evolve independently.**

### 3.2 The `Reasoner` interface (sketch)

```python
# strands_robots/reasoners/base.py
class Reasoner(ABC):
    @abstractmethod
    async def reason(
        self,
        observation: dict[str, Any],   # cameras + state, same shape as Policy obs
        question: str,
        return_thinking: bool = False,
    ) -> ReasonerResult: ...

    @abstractmethod
    async def caption_video(self, video_path: str | Path, prompt: str = "") -> str: ...

    @abstractmethod
    async def analyze_failure(
        self,
        episode_path: str | Path,      # LeRobotDataset episode dir
        task: str,
    ) -> FailureAnalysis: ...
```

`CosmosReasoner` becomes a 50-line wrapper over `strands_cosmos.CosmosVisionModel`,
plus an `agent.tool` registration so an LLM agent can reason about a robot.

### 3.3 The `WorldModel` interface (sketch)

```python
# strands_robots/world_model/base.py
class WorldModel(ABC):
    @abstractmethod
    def rollout(
        self,
        initial_observation: dict[str, Any],
        action_plan: np.ndarray,         # (T, action_dim)
        prompt: str | None = None,
    ) -> WorldModelRollout:               # contains video, predicted obs sequence
        ...

    @abstractmethod
    def augment_episode(
        self,
        episode_path: str | Path,        # LeRobotDataset episode
        prompts: list[str],              # "snowy day", "night", "foggy"
    ) -> list[Path]:                     # paths to augmented videos
        ...
```

`CosmosPredictWorldModel.rollout()` calls `cosmos_predict_generate` from
strands-cosmos. `CosmosTransferAugmentor.augment_episode()` calls
`cosmos_transfer_generate` per episode.

### 3.4 The `CosmosPolicy` (Policy provider)

Cosmos-Predict2.5 ships a `robot/policy` checkpoint post-trained on Libero +
RoboCasa. It takes `(state, image, language)` and produces action chunks —
**byte-for-byte the same contract as our existing `Gr00tPolicy`**.

So `CosmosPolicy` is structurally a clone of `policies/groot/policy.py`,
different in:
- Inference backend: HF Transformers + Cosmos-Predict2.5 wheel (or vLLM)
- Embodiment configs: Libero, RoboCasa, custom user-provided ones
- Action decoding: rectified-flow sampler (vs. GR00T's diffusion / FM)

Add to `registry/policies.json`:

```json
{
  "cosmos_predict": {
    "module": "strands_robots.policies.cosmos_predict",
    "class": "CosmosPolicy",
    "aliases": ["cosmos", "predict2.5", "cosmos-predict"],
    "requires_extras": ["cosmos"],
    "requires_trust_remote_code": true
  }
}
```

Then users do:
```python
policy = create_policy("cosmos_predict", model_id="nvidia/Cosmos-Predict2.5-2B",
                      variant="robot/policy", data_config="libero")
```

---

## 4. Dependency strategy

### 4.1 What goes in `pyproject.toml`

ADD a new optional-extras group:

```toml
[project.optional-dependencies]
cosmos = [
    "strands-cosmos>=0.2.0,<0.3.0",
    # NOTE: strands-cosmos pulls transformers, accelerate, torch, etc.
    # We do NOT add cosmos-predict2 / cosmos-transfer2 wheels here —
    # those are NVIDIA repos installed via:
    #   git clone + pip install -e .  (mirrors strands-cosmos's `just setup`)
    # Reason: the Cosmos repos are not on PyPI as wheels, they expect
    # a checkout-based install with CUDA-specific torch.
]

cosmos-edge = [
    "strands-robots[cosmos]",
    # Jetson-only extras — TRT engines, etc.
    # Mirrors strands-cosmos's [jetson] extra.
]

all = [
    "strands-robots[groot-service]",
    "strands-robots[lerobot]",
    "strands-robots[sim-mujoco]",
    "strands-robots[cosmos]",   # ADD
]
```

### 4.2 What does NOT get vendored

Following strands-cosmos's `UPSTREAM UNTOUCHED` principle:

- `cosmos-predict2.5` — installed alongside via `pip install -e ../cosmos-predict2.5`
- `cosmos-transfer2.5` — same
- `cosmos-reason1` — already wrapped by strands-cosmos's `CosmosVisionModel`

Provide a `scripts/setup_cosmos.sh` that mirrors `just setup` from strands-cosmos:

```bash
#!/usr/bin/env bash
# strands-robots/scripts/setup_cosmos.sh
set -e
cd "$(dirname "$0")/../.."  # repo parent
[ -d cosmos-predict2.5 ] || git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
[ -d cosmos-transfer2.5 ] || git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-predict2.5 && pip install -e . && cd ..
cd cosmos-transfer2.5 && pip install -e . && cd ..
echo "Cosmos environment ready. Run: pytest tests_integ/test_cosmos_*.py"
```

### 4.3 Version compatibility matrix

Lessons from strands-cosmos AGENTS.md:
- `transformers>=4.57.0,<5.3.0` (pinned to avoid `StopIteration` in `get_rope_index`)
- `torch` — must match Jetson JetPack's CUDA version on edge
- `lerobot>=0.5.0,<0.6.0` (already pinned in strands-robots)

**These pins must agree.** If strands-cosmos pins `transformers<5.3.0` and
strands-robots's lerobot dep wants `transformers>=5.3.0`, we have a conflict.
Resolution: pin both, with strands-cosmos's bound being the binding one when
`[cosmos]` extra is installed.

---

## 5. Concrete user stories

### Story 1: "Use Cosmos as my VLA on a real robot"
```python
from strands_robots import Robot, create_policy

robot = Robot("so100", mode="real")
policy = create_policy("cosmos_predict",
                      model_id="nvidia/Cosmos-Predict2.5-2B",
                      variant="robot/policy",
                      data_config="libero")
robot.run_policy(policy, instruction="pick up the red block")
```
Reuses **all** existing `Robot` infrastructure. Only `cosmos_predict` is new.

### Story 2: "Reason about a scene before acting"
```python
from strands_robots import Robot
from strands_robots.reasoners import create_reasoner

robot = Robot("so100", mode="sim")
reasoner = create_reasoner("cosmos_reason", model_id="nvidia/Cosmos-Reason2-2B")

obs = robot.get_observation()
plan = await reasoner.reason(obs, "What objects are graspable and in what order should I pick them?")
# plan is text; agent layer turns it into instructions for policy.run
```

### Story 3: "Augment my LeRobot dataset with Transfer2.5"
```python
from strands_robots.augmentation import CosmosTransferAugmentor

aug = CosmosTransferAugmentor(model_id="nvidia/Cosmos-Transfer2.5-2B",
                              control="depth")
new_paths = aug.augment_episode(
    "~/datasets/my_so100_demos/episode_0001",
    prompts=["kitchen at night", "office with bright sunlight", "workshop, dusty"],
)
# Each prompt -> 1 new photorealistic video, same actions/states
```

### Story 4: "Dream-rollout for offline policy eval"
```python
from strands_robots.world_model import create_world_model

wm = create_world_model("cosmos_predict", variant="robot/action-cond")
rollout = wm.rollout(
    initial_observation=obs,
    action_plan=policy.get_actions_sync(obs, "pick up cube"),
)
# rollout.video -> what Cosmos thinks will happen
# Useful for: failure prediction, eval-without-hardware, MPC
```

### Story 5: "Strands Agent uses everything together"
```python
from strands import Agent
from strands_robots.tools import (
    lerobot_camera, gr00t_inference,         # existing
    cosmos_reason, cosmos_dream_rollout,     # new
)

agent = Agent(tools=[lerobot_camera, gr00t_inference, cosmos_reason, cosmos_dream_rollout])
agent("Look at the table. Pick up the cube only if you predict the rollout will succeed.")
```
This is the **headline story**: an LLM agent that uses Cosmos VLM for perception,
a Cosmos world model to dream-rollout candidate plans, and an existing policy
(GR00T or Cosmos-Predict) for execution.

---

## 6. Risks and open questions

### 6.1 Risks

1. **Dep weight**: Cosmos pulls torch, transformers, accelerate, av, qwen-vl-utils,
   etc. Putting it under `[cosmos]` extras keeps `import strands_robots` cheap, but
   anyone in `[all]` pays the cost. **Mitigation**: keep `all` excluding `cosmos`,
   provide `[full]` that includes it.
2. **CUDA version drift**: Cosmos repos pin specific torch+CUDA. Jetson needs
   different wheels than desktop. **Mitigation**: `cosmos-edge` extra + the
   `strands-cosmos-fix-cublas` script; document hardware matrix.
3. **NVIDIA-only**: Cosmos is CUDA-only. **Mitigation**: this is fine — strands-robots
   already has GPU-only paths (`groot-service`); we're adding another optional one.
4. **`trust_remote_code`**: Cosmos models use HF custom code. Our existing
   `_HF_REMOTE_CODE_PROVIDERS` gate already handles this — just add `cosmos_predict`
   to that frozenset.
5. **API churn**: Cosmos repos are pre-1.0 (2.5 in flux, weekly news entries).
   **Mitigation**: pin strands-cosmos versions, defer to it for upstream-API
   absorption.

### 6.2 Open questions

1. **Should `Reasoner` and `WorldModel` live in strands-robots or strands-cosmos?**
   Argument for strands-robots: they consume robot observations, produce robot-
   meaningful outputs. Argument for strands-cosmos: they're general physical-AI.
   **My take**: ABCs in strands-robots (`reasoners/base.py`, `world_model/base.py`),
   Cosmos *implementations* in strands-robots' `cosmos_reason/` and `cosmos/`
   subfolders, USING strands-cosmos as the runtime backend. This keeps strands-
   cosmos's surface focused on Cosmos-itself and lets future non-Cosmos reasoners
   (e.g. a `LlavaReasoner`, `Pi0Reasoner`) plug in cleanly.

2. **Do we adopt the `justfile` pattern from strands-cosmos?** Today
   strands-robots is pure-Python with `hatch run`. The justfile pattern is great
   for Cosmos's TRT pipeline but overkill for a Python lib. **Recommendation**:
   no. Keep `hatch run` as truth. Cosmos pipelines stay in strands-cosmos's
   justfile; strands-robots calls them via `subprocess` only when needed.

3. **GR00T N1.7 already uses Cosmos-Reason2 as backbone.** Per
   `policies/groot/policy.py`, N1.7 detection probes for `gr00t.model.gr00t_n1d7`.
   Question: do we need a separate `CosmosPolicy` if GR00T N1.7 already wraps it?
   **Answer**: yes — Cosmos-Predict2.5/robot/policy is a *different* model family
   from GR00T-N1.7 (rectified flow vs. GR00T's stack), trained on different
   datasets (Libero/RoboCasa vs. GR00T's training mix). They are siblings, not
   the same thing.

4. **Where do training pipelines live?** Cosmos has rich post-training (LoRA,
   distillation, SFT) that strands-cosmos already wraps. **Recommendation**:
   strands-robots adds *no* training tools — defer to strands-cosmos's existing
   `cosmos_post_train`, `cosmos_distill`. We expose them as agent tools via
   `from strands_cosmos import cosmos_post_train` and that's it.

5. **Multi-agent / Zenoh story.** PR #101 brings zenoh peer-to-peer. Cosmos
   inference is heavy — perfect candidate for a *server peer*. One workstation
   peer runs Cosmos-Predict2.5 inference; multiple Jetson peers running
   strands-robots send obs over zenoh and receive actions. This is a *future*
   doc, but worth noting that the infra already lines up.

---

## 7. Phased plan

### Phase 0 — RFC (this doc)
- Land this research doc in `research/COSMOS_INTEGRATION.md`
- Open GitHub issue on the project board with this as the body
- Get review from team before writing any code

### Phase 1 — `CosmosPolicy` (lowest-risk wedge)
- New file: `strands_robots/policies/cosmos_predict/{__init__.py,policy.py,client.py,data_config.py}`
- Add `[cosmos]` extras
- Add registry entry
- Add to `_HF_REMOTE_CODE_PROVIDERS`
- Unit tests with mocked HF inference
- Integration test (gated on GPU + Cosmos-Predict2.5 weights) in `tests_integ/`
- Docs: `docs/policies/cosmos.md`

**Acceptance**: `create_policy("cosmos_predict", ...)` works on Libero sim. 
**Effort**: 1–2 weeks. 
**Owner**: TBD.

### Phase 2 — `Reasoner` interface + `CosmosReasoner`
- New subpackage `strands_robots/reasoners/`
- `base.py`, `factory.py`, `cosmos_reason/reasoner.py`
- `@tool` wrapper `cosmos_reason_tool` that takes obs+question, returns text
- Integration with `Robot` (e.g. `robot.reasoner = create_reasoner(...)`)
- Tests

**Acceptance**: a Strands Agent can do `"What is on the table?"` against a `Robot`. 
**Effort**: 1 week.

### Phase 3 — `WorldModel` interface + `CosmosPredictWorldModel`
- New subpackage `strands_robots/world_model/`
- `base.py`, `factory.py`, `cosmos/predict.py`
- Dream-rollout API
- Tests

**Acceptance**: rollout call produces video + predicted obs. 
**Effort**: 2 weeks (the hardest, depends on Cosmos repo install ergonomics).

### Phase 4 — Transfer2.5 dataset augmentation
- `strands_robots/augmentation/cosmos_transfer.py`
- LeRobotDataset round-trip (load episode → augment → write new dataset)
- Distilled-edge support for Jetson

**Acceptance**: 1 episode → N augmented variants in same LeRobotDataset format. 
**Effort**: 1–2 weeks.

### Phase 5 — Headline demo
- Notebook / example script showing **all four pieces** wired up:
  reason → dream → policy → execute → augment dataset
- Blog post / video demo

---

## 8. What I'd NOT do (tempting traps)

1. **Don't fork or vendor any Cosmos repo into strands-robots.** They're huge,
   licenses are clean (Apache-2.0) but vendoring kills upstream-tracking.
2. **Don't put Cosmos in core deps.** Optional extras only.
3. **Don't merge `Policy` and Cosmos's `robot/policy`** into a special class.
   Cosmos-Predict's policy is *just another `Policy`*. Don't special-case.
4. **Don't bypass strands-cosmos.** It already exists, it's on PyPI, you wrote it.
   Use it as a dep, don't reimplement its 21 tools in strands-robots.
5. **Don't introduce new ABCs prematurely.** `Reasoner` and `WorldModel` should
   only land when there's a *second* implementation in sight. Today there's
   only Cosmos-Reason2 and Cosmos-Predict2.5. So: ship `CosmosReasoner` and
   `CosmosPredictWorldModel` *first*, refactor into ABCs *second*. (Phase 2/3
   above already follows this; just flagging.)
6. **Don't replace `dataset_recorder.py`.** Transfer2.5 is *augmentation*, not
   replacement. It runs offline over recorded episodes, not in the recording loop.

---

## 9. Recommendation

**Start with Phase 1 (`CosmosPolicy`).** It's the lowest-risk, highest-value
wedge:
- Adds one provider, ~300 LOC, follows the `Gr00tPolicy` template exactly
- Unlocks Libero/RoboCasa benchmarking with Cosmos checkpoints
- Validates the dependency strategy before committing to bigger surface
- Doesn't require new ABCs — just slots into existing `policies/` registry

If Phase 1 lands cleanly, Phase 2–5 follow with confidence.

If it doesn't land cleanly (CUDA hell, HF-version conflicts), we learn that
*before* we've written the `Reasoner`/`WorldModel` ABCs.

---

## 10. Appendix: file-level deltas if we ship Phase 1

```
# pyproject.toml
+[project.optional-dependencies]
+cosmos = ["strands-cosmos>=0.2.0,<0.3.0"]
+all = [..., "strands-robots[cosmos]"]

# strands_robots/registry/policies.json
+"cosmos_predict": {
+  "module": "strands_robots.policies.cosmos_predict",
+  "class": "CosmosPolicy",
+  "aliases": ["cosmos", "predict2.5", "cosmos-predict"],
+  "requires_extras": ["cosmos"]
+}

# strands_robots/policies/factory.py
 _HF_REMOTE_CODE_PROVIDERS: frozenset[str] = frozenset({
     "lerobot_local",
+    "cosmos_predict",
 })

# NEW FILES
strands_robots/policies/cosmos_predict/__init__.py        ~10 LOC
strands_robots/policies/cosmos_predict/policy.py          ~250 LOC (clone of groot/policy.py shape)
strands_robots/policies/cosmos_predict/client.py          ~150 LOC (HF inference)
strands_robots/policies/cosmos_predict/data_config.py     ~80 LOC (Libero/RoboCasa configs)
strands_robots/policies/cosmos_predict/data_configs.json  ~50 LOC (JSON of configs)
tests/policies/test_cosmos_predict.py                     ~120 LOC (mocked)
tests_integ/test_cosmos_predict_integ.py                  ~80 LOC (real model)
docs/policies/cosmos.md                                   prose
scripts/setup_cosmos.sh                                   ~20 LOC
```

Approx **~1000 LOC + tests + 1 doc page** to ship Phase 1.

---

## 11. Action items (if approved)

- [ ] Open issue on https://github.com/orgs/strands-labs/projects/2 with this doc as body, Status=`Backlog`, Priority=`Medium`
- [ ] Land this `research/COSMOS_INTEGRATION.md` on `main` (research only — no code)
- [ ] Get +1 from team on the three-abstraction split (Policy / Reasoner / WorldModel)
- [ ] Get +1 on the `strands-cosmos` runtime-dep strategy (vs. vendoring)
- [ ] Spike `CosmosPolicy` Phase 1 on a feature branch
- [ ] If spike passes: open PR, follow PR-review-learnings checklist from AGENTS.md

