# SOLUTION — Bulletproof Policy Observation/Action Mapping

> **Problem statement.** Our `lerobot_local` policy maps robot/sim observations
> to model feature keys using **imperative heuristics on the hot path**
> (`_to_lerobot_observation`, `_build_batch_from_strands_format`,
> `_fixup_preprocessed_batch`, `_build_batch_from_lerobot_format`). These:
> 1. run **every inference step** (string `"image" in key` scans, `ndim>=2`
>    image-guessing, positional slot-fill, per-key np→torch conversions,
>    dim pad/truncate) → wasted CPU on the control loop;
> 2. **silently mismap** (positional fill when names don't match, generic
>    `joint_0..N` keys that never match real sim keys `'1'..'6'`, truncation
>    that drops state) → the policy often gets structurally wrong inputs and
>    produces bad actions.
>
> **Root cause:** we reinvented a mapping layer that LeRobot already ships as a
> **declarative, serializable, registered pipeline step** — and which we are
> currently bypassing.

---

## 1. The smoking gun

### 1a. LeRobot's canonical mapping is declarative JSON — and we ignore it

Every LeRobot policy preprocessor pipeline **starts** with a rename step:

```python
# lerobot/policies/{smolvla,pi05,molmoact2,groot,gaussian_actor}/processor_*.py
input_steps = [
    RenameObservationsProcessorStep(rename_map={}),   # ← FIRST step, ALWAYS
    AddBatchDimensionProcessorStep(),
    NormalizerProcessorStep(...),
    TokenizerProcessorStep(...),
    DeviceProcessorStep(...),
    ...
]
```

`RenameObservationsProcessorStep` (`lerobot/processor/rename_processor.py`) is:
- **declarative**: `rename_map: dict[str, str]` (old_key → new_key);
- **serializable**: `get_config() -> {"rename_map": ...}` (round-trips through
  `preprocessor.json`);
- **registered**: `@ProcessorStepRegistry.register("rename_observations_processor")`;
- **feature-aware**: `transform_features()` keeps the policy's declared
  `input_features` consistent with the rename.

LeRobot ships it **empty** (`rename_map={}`) on purpose — the env→model key map
is deployment-specific. It is the **designated extension point** for exactly our
problem. We never populate it. Instead we do the rename ourselves, *before*
handing obs to the pipeline, with heuristics.

### 1b. Action mapping has the same canonical pair (also ignored)

`lerobot/processor/policy_robot_bridge.py`:
```python
@ProcessorStepRegistry.register("policy_action_to_robot_action_processor")
class PolicyActionToRobotActionProcessorStep:
    motor_names: list[str]
    def action(self, action): return {f"{name}.pos": action[i]
                                       for i, name in enumerate(self.motor_names)}
```
A declarative `motor_names` list maps the action tensor → named robot dict. Our
`_tensor_to_action_dicts` reimplements this with `robot_state_keys` indexing and
a silent `0.0` fallback on length mismatch.

### 1c. We already proved the declarative pattern works — twice, internally

- **GR00T** (`policies/groot/policy.py`): `ObservationMapping` / `ActionMapping`
  frozen dataclasses, **validated against model modality configs**, auto-inferred
  from `data_configs.json` (exact-name match → positional fallback **with a
  WARNING**, never silent). This is the correct shape.
- **`data_configs.json`** + `Gr00tDataConfig` with `_extends` inheritance: a
  declarative, per-embodiment key registry. Exactly what `lerobot_local` lacks.

So the fix is not invention — it is **applying the pattern we already use for
GR00T to `lerobot_local`, backed by LeRobot's own rename step.**

### 1d. Why the current heuristics are fragile (concrete failure modes)

From `MUJOCO_FINDINGS.md`, every one of these was a real bug born from
heuristic mapping:

| Bug | Heuristic that caused it |
|-----|--------------------------|
| **B12** | sim cameras `image`/`wrist_image` not namespaced to `observation.images.*` → preprocessor `image_keys missing` |
| **B12b** | state-dim guessed, pad/truncate to model dim **inside the hot path** |
| **B12c** | `robot_state_keys` auto-filled `joint_0..6` never matched sim keys `'1'..'6'` → fell back to obs scalars |
| general | `"image" in key` substring test; `ndim>=2 ⇒ image`; positional image slot-fill; `float64→float32` per key |

Each was patched with *another* special case. The heuristic layer is now ~250
lines across 4 methods, all executing per step, and still embodiment-blind.

---

## 2. The solution: a declarative `EmbodimentMap`, applied once at load

### 2.1 Principle

> **Build the obs/action mapping ONCE at policy-load time from a declarative
> spec, inject it into LeRobot's own `RenameObservationsProcessorStep` (and the
> action-bridge step), validate it against the model's declared features, then
> let the pipeline do 100% of the per-step work. The hot path becomes
> `obs → pipeline → policy → pipeline → action` with ZERO strands-side
> remapping.**

### 2.2 The declarative spec

Add an `EmbodimentMap` (mirrors `Gr00tDataConfig`) — source it from a JSON
registry keyed by `(robot, model)` or `data_config`, with `_extends` inheritance:

```python
@dataclass(frozen=True)
class EmbodimentMap:
    """Declarative robot/sim ↔ model key mapping. Built once, validated once."""
    # sim/robot obs key  →  model feature key
    obs_rename: dict[str, str]          # {"image": "observation.images.image",
                                        #  "wrist_image": "observation.images.wrist_image"}
    # ordered sim joint keys that compose observation.state
    state_keys: list[str]               # ["1","2","3","4","5","6","gripper"]
    # ordered robot actuator names for the action tensor
    action_keys: list[str]              # ["1","2","3","4","5","6","gripper"]
    # how to reconcile a state/action dim mismatch vs the model (explicit, not silent)
    dim_policy: str = "strict"          # "strict" | "pad" | "truncate"
```

Stored declaratively (extends the existing registry pattern):

```jsonc
// strands_robots/policies/lerobot_local/embodiments.json
{
  "configs": {
    "panda_libero": {
      "obs_rename": {
        "image":        "observation.images.image",
        "wrist_image":  "observation.images.wrist_image"
      },
      "state_keys":  ["x","y","z","roll","pitch","yaw","gripper"],
      "action_keys": ["x","y","z","roll","pitch","yaw","gripper"],
      "dim_policy":  "strict"
    },
    "so101": {
      "obs_rename": { "front": "observation.images.image",
                      "wrist": "observation.images.wrist_image" },
      "state_keys":  ["1","2","3","4","5","6"],
      "action_keys": ["1","2","3","4","5","6"],
      "dim_policy":  "pad"          // 6→8 LIBERO, explicit & logged ONCE
    }
  },
  "aliases": { "franka_libero": "panda_libero" }
}
```

### 2.3 Where the map plugs in (one-time, at `_load_model`)

LeRobot pipelines accept **`overrides`** in `from_pretrained` keyed by registry
step name. We already pass `overrides` through `ProcessorBridge.from_pretrained`.
So we inject the rename map straight into the model's existing rename step:

```python
# in ProcessorBridge.from_pretrained / LerobotLocalPolicy._load_model
overrides = {
    "rename_observations_processor": {"rename_map": embodiment.obs_rename},
    # state vector composition handled by a tiny PackStateProcessorStep (below)
    # inserted right after rename, BEFORE the normalizer.
}
```

For composing scalar joint keys → `observation.state` we add ONE small
registered step (the only new pipeline code), inserted after rename:

```python
@ProcessorStepRegistry.register("strands_pack_state")
@dataclass
class PackStateProcessorStep(ObservationProcessorStep):
    state_keys: list[str]
    expected_dim: int
    dim_policy: str = "strict"
    def observation(self, obs):
        if "observation.state" in obs:           # already packed → passthrough
            return obs
        vals = [float(obs[k]) for k in self.state_keys if k in obs]
        vals = _reconcile_dim(vals, self.expected_dim, self.dim_policy)  # explicit
        out = {k: v for k, v in obs.items() if k not in self.state_keys}
        out["observation.state"] = np.asarray(vals, dtype=np.float32)
        return out
```

`expected_dim` is read **once** from `config.input_features["observation.state"].shape`.
`_reconcile_dim` raises on `strict`, pads/truncates with a **single** log line on
`pad`/`truncate` — no per-step warnings, no silent corruption.

### 2.4 Action side (symmetric, declarative)

Replace `_tensor_to_action_dicts`'s positional indexing with the embodiment's
`action_keys` (validated length == model action dim at load). Optionally use
LeRobot's `PolicyActionToRobotActionProcessorStep(motor_names=action_keys)` in
the postprocessor so the unnormalizer + naming live in one pipeline.

### 2.5 Validation at load (fail-fast, like GR00T)

```python
def validate(self, input_features, output_features):
    # every rename target must be a declared model feature
    for src, dst in self.obs_rename.items():
        if dst not in input_features:
            raise ValueError(f"obs_rename {src!r}→{dst!r} not in model "
                             f"input_features {sorted(input_features)}")
    sdim = input_features["observation.state"].shape[0]
    if self.dim_policy == "strict" and len(self.state_keys) != sdim:
        raise ValueError(f"state_keys has {len(self.state_keys)} keys but model "
                         f"expects {sdim}; set dim_policy='pad'/'truncate' to opt in")
    adim = output_features["action"].shape[0]
    if len(self.action_keys) != adim:
        raise ValueError(f"action_keys {len(self.action_keys)} != model action dim {adim}")
```

A misconfigured embodiment fails **at load with a precise message**, instead of
silently feeding garbage for an entire rollout.

---

## 3. Resulting hot path (before → after)

**Before** (per step):
```
get_actions(obs)
 ├─ dict(obs)                                   # copy
 ├─ _to_lerobot_observation(obs)                # scan keys, "image" substring,
 │    ├─ declared_img_feats list-comp           #   ndim>=2 guess, positional
 │    ├─ exact/positional image slot fill       #   slot fill, dim pad/truncate,
 │    └─ state collection + dim adapt            #   np.asarray
 ├─ preprocess(lerobot_obs)                      # pipeline (rename={} no-op)
 ├─ _fixup_preprocessed_batch(batch)            # re-scan EVERY key: np→torch,
 │                                               #   HWC→CHW, unsqueeze, .to(dev)
 └─ select_action(batch)
```

**After** (per step):
```
get_actions(obs)
 └─ preprocess(obs)        # rename + pack_state + batch + normalize + device
                           #   ALL declarative, built once, C-fast tensor ops
 └─ select_action(batch)
```

- `_to_lerobot_observation`, `_fixup_preprocessed_batch`,
  `_build_batch_from_strands_format`, `_build_batch_from_lerobot_format` →
  **deleted** (or kept only as a no-pipeline fallback for raw policies w/o a
  preprocessor, clearly marked legacy).
- Per-step Python key-scanning/string-matching → **gone**. Shape/device/dtype
  handled by `AddBatchDimensionProcessorStep` + `DeviceProcessorStep` which
  already run in the pipeline.

---

## 4. Why this is bulletproof

| Property | Heuristic (now) | Declarative `EmbodimentMap` (proposed) |
|----------|-----------------|-----------------------------------------|
| Correctness | guesses by substring/shape/position | exact named map, **validated vs model features** |
| Failure mode | silent mismap / truncate | **fail-fast at load** with precise error |
| Hot-path cost | full key scan + np conv every step | zero strands code; pipeline-only |
| Embodiment-aware | no (generic `joint_0..N`) | yes (per-robot JSON, `_extends`) |
| Uses LeRobot's design | bypasses `rename_map` | **populates** the step LeRobot ships for this |
| Round-trips to disk | no | yes (`get_config()` → `preprocessor.json`) |
| Mirrors existing code | — | same shape as GR00T `ObservationMapping` |
| Testable | mocks hide it (B7/B12 slipped past) | map is data → unit-test the JSON + validate() |

---

## 5. Implementation plan (incremental, no big-bang)

1. **Add** `embodiments.json` + `EmbodimentMap` loader (`_extends`, aliases) —
   copy `groot/data_config.py` structure. *(pure data, no behaviour change)*
2. **Add** `PackStateProcessorStep` (registered) + `_reconcile_dim` helper.
3. **Wire** `LerobotLocalPolicy.__init__(embodiment=...)`; in `_load_model`
   build overrides `{rename_observations_processor: {rename_map}, + insert
   strands_pack_state}` and call `embodiment.validate(input,output)`.
4. **Route** `get_actions`: when a preprocessor + embodiment exist, feed **raw**
   obs straight to `preprocess()` (pipeline now renames+packs). Drop the
   `_to_lerobot_observation` + `_fixup_preprocessed_batch` calls on this path.
5. **Action**: replace positional `_tensor_to_action_dicts` with
   `embodiment.action_keys` (or postprocessor bridge step).
6. **Deprecate** the four heuristic builders → keep only as the
   `preprocessor is None` raw fallback, labelled legacy.
7. **Tests**: (a) `embodiments.json` schema + `validate()` raises on bad dim /
   unknown feature; (b) a real `preprocess()` run asserts `rename_map` is
   populated and `observation.state` packed in the right order; (c) regression:
   MolmoAct2-Panda-LIBERO rollout still produces the same monotonic trajectory
   from `MUJOCO_FINDINGS.md` — now via the declarative path. These close the
   "mocks hid B7/B12" gap by exercising the **real** mapping.

### Backward compatibility
- Raw policies with no preprocessor.json keep working via the legacy fallback.
- `robot_state_keys=` ctor arg stays as a shorthand that auto-builds a trivial
  `EmbodimentMap(state_keys=robot_state_keys, action_keys=robot_state_keys)`.
- Existing GR00T path untouched (it already has its own mapping).

---

## 6. One-line summary

**Stop remapping observations imperatively on the control loop. Populate
LeRobot's own `RenameObservationsProcessorStep.rename_map` (+ a tiny registered
`PackStateProcessorStep`) ONCE at load from a declarative, validated,
per-embodiment JSON — exactly the pattern GR00T already uses — and let the
pipeline own every per-step transform.**
