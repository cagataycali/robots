# RFC: Eliminate Fragile Key Mapping in Gr00tPolicy

**Author:** DevDuck / Cagatay  
**Date:** 2026-03-17  
**Status:** Draft — next PR after feat/groot-upgrade merges  
**Tracks:** [strands-labs/robots#55](https://github.com/strands-labs/robots/pull/55)

---

## Problem Statement

`Gr00tPolicy` currently has **three separate key namespaces** that collide at inference time:

| Layer | Example Keys | Where It Lives |
|-------|-------------|----------------|
| **Robot keys** | `ego_view`, `joint_position`, `gripper` | Raw sensor data from the physical robot |
| **Our data config keys** | `video.webcam`, `state.single_arm`, `annotation.human.task_description` | `_data_config_defs.py` |
| **Model's modality keys** | `video.ego_view_bg_crop_pad_res256_freq20`, `state.left_arm`, `task` | Isaac-GR00T `processor.get_modality_configs()[embodiment_tag]` |

The current pipeline does **two blind positional remaps**:

```
Robot keys ──(_build_observation)──> Our data config keys ──(_local_inference_n16)──> Model modality keys
```

Both remaps are **positional** (zip the lists together), which means:
1. If order changes in any list, mappings silently break
2. If key counts differ, we zero-fill with guessed dimensions
3. There's no way to know *at construction time* if a mapping is wrong
4. Adding a new robot requires getting the list ordering exactly right

### Concrete Failures We've Seen

1. **SO-100 → GR1 model**: Our data config has `video.webcam`, but the GR1 model expects `video.ego_view_bg_crop_pad_res256_freq20`. Positional remap happens to work because both have exactly 1 video key — but it's a coincidence.

2. **State key count mismatch**: SO-100 has 2 state groups (`single_arm`, `gripper` = 6 DOF), but GR1 model expects 5 state groups (`left_arm:7`, `right_arm:7`, `left_hand:6`, `right_hand:6`, `waist:3` = 30 DOF). We zero-fill 4 out of 5 state groups — the model gets mostly zeros.

3. **Language key mismatch**: Our data config uses `annotation.human.task_description` but GR1 model expects just `task`. The `Gr00tSimPolicyWrapper` has a hardcoded patch for `annotation.human.coarse_action` → `task`, but nothing for our key format.

---

## What the Model Actually Expects (Ground Truth)

Extracted from `GR00TProcessor.from_pretrained("nvidia/GR00T-N1.6-3B").get_modality_configs()`:

```
behavior_r1_pro:
  video: keys=['observation.images.rgb.head_256_256', 'observation.images.rgb.left_wrist_256_256', 'observation.images.rgb.right_wrist_256_256']
  state: keys=['robot_pos', 'robot_ori_cos', 'robot_ori_sin', 'robot_2d_ori', 'robot_2d_ori_cos', 'robot_2d_ori_sin', 'robot_lin_vel', 'robot_ang_vel', 'arm_left_qpos', 'arm_left_qpos_sin', 'arm_left_qpos_cos', 'eef_left_pos', 'eef_left_quat', 'gripper_left_qpos', 'arm_right_qpos', 'arm_right_qpos_sin', 'arm_right_qpos_cos', 'eef_right_pos', 'eef_right_quat', 'gripper_right_qpos', 'trunk_qpos']
  action: keys=['base', 'torso', 'left_arm', 'left_gripper', 'right_arm', 'right_gripper']
  language: keys=['annotation.human.coarse_action']

gr1:
  video: keys=['ego_view_bg_crop_pad_res256_freq20']
  state: keys=['left_arm', 'right_arm', 'left_hand', 'right_hand', 'waist']
  action: keys=['left_arm', 'right_arm', 'left_hand', 'right_hand', 'waist']
  language: keys=['task']

robocasa_panda_omron:
  video: keys=['res256_image_side_0', 'res256_image_side_1', 'res256_image_wrist_0']
  state: keys=['end_effector_position_relative', 'end_effector_rotation_relative', 'gripper_qpos', 'base_position', 'base_rotation']
  action: keys=['end_effector_position', 'end_effector_rotation', 'gripper_close', 'base_motion', 'control_mode']
  language: keys=['annotation.human.action.task_description']
```

### Key Insight from `Gr00tSimPolicyWrapper._get_action()`

The wrapper does this exact transformation (line 596-612 of Isaac-GR00T's gr00t_policy.py):

```python
# It expects flat keys like 'video.ego_view_bg_crop_pad_res256_freq20'
# It converts to nested format: {'video': {'ego_view_bg_crop_pad_res256_freq20': arr}}
for modality in ["video", "state", "language"]:
    new_obs[modality] = {}
    for key in self.policy.modality_configs[modality].modality_keys:
        if modality == "language":
            parsed_key = key  # language keys are NOT prefixed
        else:
            parsed_key = f"{modality}.{key}"  # video/state keys ARE prefixed
        arr = observation[parsed_key]
        new_obs[modality][key] = arr
```

**The contract is clear:** pass flat keys that are `f"{modality}.{model_key}"` for video/state, and the raw model language key for language. No room for our own naming convention.

---

## Proposed Solution: Explicit Observation/Action Mapping

### Core Idea

Replace the implicit positional remap with an **explicit, user-declared mapping** that directly connects robot sensor keys to model modality keys. Eliminate the intermediate "our data config keys" namespace entirely for the inference path.

### New Constructor Signature

```python
policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.6-3B",
    embodiment_tag="GR1",
    
    # NEW: Explicit mapping from robot → model
    observation_mapping={
        # robot_key → model flat key
        "ego_view":       "video.ego_view_bg_crop_pad_res256_freq20",
        "joint_position": "state.left_arm",
        "gripper":        "state.left_hand",
    },
    action_mapping={
        # model flat key → robot_key
        "action.left_arm":  "joint_position",
        "action.left_hand": "gripper",
    },
    language_key="task",  # model's language key
)
```

### Data Structures

```python
@dataclass
class ObservationMapping:
    """Declares how robot sensor names map to model modality keys."""
    
    # robot_camera_name → model video key (without 'video.' prefix)
    video: Dict[str, str]
    
    # robot_state_name → (model state key without 'state.' prefix, DOF)
    state: Dict[str, Tuple[str, int]]
    
    # The model's language key (e.g. 'task', 'annotation.human.coarse_action')
    language_key: str
    
    @classmethod
    def from_dict(cls, mapping: dict, model_modality_configs: dict) -> "ObservationMapping":
        """Create from a flat robot→model dict, validating against model configs."""
        video = {}
        state = {}
        language_key = model_modality_configs["language"].modality_keys[0]
        
        model_video_keys = set(model_modality_configs["video"].modality_keys)
        model_state_keys = set(model_modality_configs["state"].modality_keys)
        
        for robot_key, model_key in mapping.items():
            if model_key.startswith("video."):
                bare = model_key.removeprefix("video.")
                assert bare in model_video_keys, f"Model has no video key '{bare}'. Available: {model_video_keys}"
                video[robot_key] = bare
            elif model_key.startswith("state."):
                bare = model_key.removeprefix("state.")
                assert bare in model_state_keys, f"Model has no state key '{bare}'. Available: {model_state_keys}"
                # DOF can be inferred from robot data at runtime, or specified
                state[robot_key] = (bare, None)  # None = infer at runtime
            else:
                raise ValueError(f"Mapping value must start with 'video.' or 'state.': {model_key}")
        
        return cls(video=video, state=state, language_key=language_key)


@dataclass  
class ActionMapping:
    """Declares how model action keys map back to robot actuator names."""
    
    # model action key (without 'action.' prefix) → robot_actuator_name
    actions: Dict[str, str]
    
    @classmethod
    def from_dict(cls, mapping: dict) -> "ActionMapping":
        actions = {}
        for model_key, robot_key in mapping.items():
            bare = model_key.removeprefix("action.")
            actions[bare] = robot_key
        return cls(actions=actions)
```

### New `_prepare_observation()` — Single-Step Transform

```python
def _prepare_observation(
    self,
    robot_obs: Dict[str, Any],
    instruction: str,
) -> dict:
    """Transform robot observations → model-ready flat dict in one step.
    
    This replaces both _build_observation() and the remap logic in 
    _local_inference_n16(). No intermediate namespace. No positional matching.
    
    Args:
        robot_obs: Raw sensor data keyed by robot names (e.g. {'ego_view': img, 'joint_position': arr})
        instruction: Natural language task instruction
    
    Returns:
        Flat dict with model's expected keys, batched for N1.6:
        {
            'video.ego_view_bg_crop_pad_res256_freq20': np.ndarray (1, 1, H, W, C),
            'state.left_arm': np.ndarray (1, 1, 7),
            'state.right_arm': np.ndarray (1, 1, 7),  # zero-filled
            ...
            'task': ['pick up the cube'],
        }
    """
    model_mc = self._local_policy.policy.modality_configs
    batched = {}
    
    # ── Video ──
    for robot_key, model_bare_key in self._obs_mapping.video.items():
        flat_key = f"video.{model_bare_key}"
        if robot_key in robot_obs:
            batched[flat_key] = self._add_batch_temporal(flat_key, robot_obs[robot_key])
        else:
            logger.warning("Robot key '%s' not in observation — zero-filling '%s'", robot_key, flat_key)
    
    # Zero-fill model video keys that have NO robot mapping at all
    for model_bare_key in model_mc["video"].modality_keys:
        flat_key = f"video.{model_bare_key}"
        if flat_key not in batched:
            # Use first available video shape as reference, or default 256x256x3
            ref_shape = self._get_reference_video_shape(robot_obs)
            batched[flat_key] = np.zeros((1, 1, *ref_shape), dtype=np.uint8)
            logger.debug("Zero-filled unmapped video key: %s", flat_key)
    
    # ── State ──
    for robot_key, (model_bare_key, _dof) in self._obs_mapping.state.items():
        flat_key = f"state.{model_bare_key}"
        if robot_key in robot_obs:
            arr = np.asarray(robot_obs[robot_key], dtype=np.float32)
            batched[flat_key] = self._add_batch_temporal(flat_key, arr)
        else:
            logger.warning("Robot key '%s' not in observation — zero-filling '%s'", robot_key, flat_key)
    
    # Zero-fill model state keys that have NO robot mapping
    for model_bare_key in model_mc["state"].modality_keys:
        flat_key = f"state.{model_bare_key}"
        if flat_key not in batched:
            # Infer DOF from the model's training config or use a sensible default
            dof = self._get_model_state_dof(model_bare_key)
            batched[flat_key] = np.zeros((1, 1, dof), dtype=np.float32)
            logger.debug("Zero-filled unmapped state key: %s (dof=%d)", flat_key, dof)
    
    # ── Language ──
    lang_key = self._obs_mapping.language_key
    batched[lang_key] = [instruction]
    
    logger.debug("Prepared observation keys: %s", list(batched.keys()))
    return batched
```

### New `_unpack_actions()` — Single-Step Reverse Transform

```python
def _unpack_actions(self, raw_actions: dict) -> List[Dict[str, Any]]:
    """Transform model action output → per-timestep robot actuator dicts.
    
    Args:
        raw_actions: Model output like {'left_arm': (1, 16, 7), 'right_arm': (1, 16, 7), ...}
    
    Returns:
        List of dicts keyed by robot actuator names:
        [
            {'joint_position': np.array([...]), 'gripper': np.array([...])},  # t=0
            {'joint_position': np.array([...]), 'gripper': np.array([...])},  # t=1
            ...
        ]
    """
    # Squeeze batch dim: (1, H, D) → (H, D)
    squeezed = {}
    for model_key, arr in raw_actions.items():
        bare = model_key.removeprefix("action.")
        while arr.ndim > 2:
            arr = arr[0]
        squeezed[bare] = arr
    
    if not squeezed:
        return []
    
    horizon = next(iter(squeezed.values())).shape[0]
    
    actions = []
    for t in range(horizon):
        step = {}
        for model_bare_key, robot_key in self._action_mapping.actions.items():
            if model_bare_key in squeezed:
                step[robot_key] = squeezed[model_bare_key][t]
        
        # Also include unmapped action keys (model keys the user didn't map to robot)
        for model_bare_key in squeezed:
            if model_bare_key not in self._action_mapping.actions:
                step[f"unmapped.{model_bare_key}"] = squeezed[model_bare_key][t]
        
        actions.append(step)
    
    return actions
```

### DOF Inference from Model Config

The zero-filling problem (guessing DOF for missing state keys) can be solved by reading the model's training data config. Isaac-GR00T stores the DOF per state key in the modality config.

```python
# Known DOF per model state key (extracted from training configs)
# This could also be read from the model's config at load time
_MODEL_STATE_DOF = {
    # GR1
    "left_arm": 7, "right_arm": 7,
    "left_hand": 6, "right_hand": 6,
    "waist": 3,
    # Unitree G1
    "left_leg": 6, "right_leg": 6,
    # Panda
    "end_effector_position_relative": 3, "end_effector_rotation_relative": 4,
    "gripper_qpos": 1, "base_position": 3, "base_rotation": 4,
    # SO-100
    "single_arm": 5, "gripper": 1,
}

def _get_model_state_dof(self, model_bare_key: str) -> int:
    """Get the DOF for a model state key."""
    # Try known DOF map first
    if model_bare_key in _MODEL_STATE_DOF:
        return _MODEL_STATE_DOF[model_bare_key]
    # Try to infer from model's actual config (if we ran an inference before)
    if hasattr(self, '_observed_state_dof') and model_bare_key in self._observed_state_dof:
        return self._observed_state_dof[model_bare_key]
    # Fallback
    logger.warning("Unknown DOF for model state key '%s' — defaulting to 6", model_bare_key)
    return 6
```

**Better approach:** At model load time, run a single dummy inference to discover DOF:

```python
def _discover_model_state_dof(self):
    """Discover DOF per state key from the model at load time.
    
    N1.6 models have the DOF baked into the modality config's normalizer.
    We can extract it without running inference.
    """
    mc = self._local_policy.policy.modality_configs
    self._model_state_dof = {}
    
    # Try reading from normalizer stats (which have the exact DOF)
    normalizer = getattr(self._local_policy.policy, 'normalizer', None)
    if normalizer:
        for key in mc["state"].modality_keys:
            stat = normalizer.get_stat(f"state.{key}")
            if stat is not None and hasattr(stat, 'shape'):
                self._model_state_dof[key] = stat.shape[-1]
    
    # Fallback to known DOF map
    for key in mc["state"].modality_keys:
        if key not in self._model_state_dof:
            self._model_state_dof[key] = _MODEL_STATE_DOF.get(key, 6)
    
    logger.info("Discovered model state DOF: %s", self._model_state_dof)
```

---

## Auto-Inference of Mapping (Backward Compatibility)

For users who don't want to specify explicit mappings, we can auto-infer from the `data_config` + `embodiment_tag`. This preserves backward compatibility.

```python
def _auto_infer_mapping(self) -> ObservationMapping:
    """Auto-infer mapping from data_config keys → model keys.
    
    Heuristic: match by stripped key name similarity.
    e.g. 'state.left_arm' in our config → 'left_arm' in model state keys
    """
    model_mc = self._local_policy.policy.modality_configs
    
    # Video: match by position (usually 1-to-1)
    video = {}
    our_video = [k.removeprefix("video.") for k in self.data_config.video_keys]
    model_video = list(model_mc["video"].modality_keys)
    for i, our_key in enumerate(our_video):
        if i < len(model_video):
            video[our_key] = model_video[i]
            if our_key != model_video[i]:
                logger.info("Auto-mapped video: %s → %s (positional)", our_key, model_video[i])
    
    # State: try exact match first, then positional fallback
    state = {}
    our_state = [k.removeprefix("state.") for k in self.data_config.state_keys]
    model_state = list(model_mc["state"].modality_keys)
    model_state_set = set(model_state)
    
    used_model_keys = set()
    # Pass 1: exact name match
    for our_key in our_state:
        if our_key in model_state_set:
            state[our_key] = (our_key, None)
            used_model_keys.add(our_key)
    # Pass 2: positional fallback for unmatched
    remaining_our = [k for k in our_state if k not in state]
    remaining_model = [k for k in model_state if k not in used_model_keys]
    for our_key, model_key in zip(remaining_our, remaining_model):
        state[our_key] = (model_key, None)
        logger.warning("Auto-mapped state: %s → %s (positional fallback — consider explicit mapping)", our_key, model_key)
    
    # Language
    language_key = model_mc["language"].modality_keys[0]
    
    return ObservationMapping(video=video, state=state, language_key=language_key)
```

---

## What Gets Deleted

With explicit mapping, these become unnecessary:

| Current Code | Replacement |
|-------------|-------------|
| `_build_observation()` | `_prepare_observation()` — single-step, no intermediate keys |
| `_map_video_key_to_camera()` + `_CAMERA_ALIASES` | Explicit `observation_mapping.video` — no guessing |
| `_map_robot_state_to_groot_state()` + `_STATE_LAYOUTS` | Explicit `observation_mapping.state` — no layout tables |
| `_local_inference_n16()` positional remap | Gone — `_prepare_observation()` outputs model-ready keys |
| `_convert_to_robot_actions()` concat+split | `_unpack_actions()` — direct key-to-key reverse mapping |
| `robot_state_keys` flat array mapping | Gone — state is already keyed by robot name |

Total lines deleted: ~150. Total lines added: ~100. Net: simpler.

---

## Migration Path

### Phase 1 (This PR — already done)
- Positional remap in `_local_inference_n16()` works for current embodiments
- All 92 unit + 8 integration tests pass

### Phase 2 (Next PR — this RFC)
1. Add `ObservationMapping` and `ActionMapping` dataclasses
2. Add `observation_mapping` / `action_mapping` params to constructor
3. Implement `_prepare_observation()` and `_unpack_actions()`  
4. Add `_auto_infer_mapping()` for backward compatibility
5. Deprecate `_build_observation()`, `_map_robot_state_to_groot_state()`, etc.
6. Update tests to use explicit mappings

### Phase 3 (Follow-up)
1. YAML-based robot profiles (see below)
2. Remove deprecated code
3. Auto-discover DOF from model normalizer

---

## YAML Robot Profiles

Long term, mappings should live in config files alongside robot definitions:

```yaml
# robot_profiles/so100_gr1.yaml
name: so100_on_gr1
description: "SO-100 arm using GR1 base model"
model: nvidia/GR00T-N1.6-3B
embodiment_tag: GR1

observation_mapping:
  webcam: video.ego_view_bg_crop_pad_res256_freq20
  joint_position: state.left_arm      # SO-100's 5 joints → GR1's left_arm (7 DOF, 2 zero-filled)
  gripper: state.left_hand             # SO-100's gripper → GR1's left_hand (6 DOF, 5 zero-filled)

action_mapping:
  action.left_arm: joint_position      # First 5 DOF → SO-100 joints
  action.left_hand: gripper            # First 1 DOF → SO-100 gripper

# Optional: explicit DOF overrides (otherwise auto-inferred from model)
state_dof:
  joint_position: 5
  gripper: 1
```

```yaml
# robot_profiles/gr1_native.yaml  
name: gr1_native
description: "Fourier GR-1 using GR1 base model (native mapping)"
model: nvidia/GR00T-N1.6-3B
embodiment_tag: GR1

observation_mapping:
  ego_view: video.ego_view_bg_crop_pad_res256_freq20
  left_arm: state.left_arm
  right_arm: state.right_arm
  left_hand: state.left_hand
  right_hand: state.right_hand
  waist: state.waist

action_mapping:
  action.left_arm: left_arm
  action.right_arm: right_arm
  action.left_hand: left_hand
  action.right_hand: right_hand
  action.waist: waist
```

Usage:
```python
from strands_robots.policies.groot import Gr00tPolicy

# From YAML profile
policy = Gr00tPolicy.from_profile("robot_profiles/so100_gr1.yaml")

# Or inline (same as YAML content)
policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.6-3B",
    embodiment_tag="GR1",
    observation_mapping={
        "webcam": "video.ego_view_bg_crop_pad_res256_freq20",
        "joint_position": "state.left_arm",
        "gripper": "state.left_hand",
    },
    action_mapping={
        "action.left_arm": "joint_position",
        "action.left_hand": "gripper",
    },
)
```

---

## Available EmbodimentTags in N1.6

For reference, these are the tags available in `nvidia/GR00T-N1.6-3B`:

| Tag | Value | Video Keys | State Keys | Action Keys | Language Key |
|-----|-------|-----------|------------|-------------|-------------|
| `GR1` | `gr1` | `ego_view_bg_crop_pad_res256_freq20` | `left_arm, right_arm, left_hand, right_hand, waist` | same as state | `task` |
| `UNITREE_G1` | `unitree_g1` | (same pattern) | (G1 joints) | (G1 joints) | varies |
| `ROBOCASA_PANDA_OMRON` | `robocasa_panda_omron` | `res256_image_side_0, res256_image_side_1, res256_image_wrist_0` | `end_effector_position_relative, end_effector_rotation_relative, gripper_qpos, base_position, base_rotation` | `end_effector_position, end_effector_rotation, gripper_close, base_motion, control_mode` | `annotation.human.action.task_description` |
| `BEHAVIOR_R1_PRO` | `behavior_r1_pro` | `observation.images.rgb.head_256_256, ...left_wrist_256_256, ...right_wrist_256_256` | 21 state keys | `base, torso, left_arm, left_gripper, right_arm, right_gripper` | `annotation.human.coarse_action` |
| `LIBERO_PANDA` | `libero_panda` | (sim) | (sim) | (sim) | varies |
| `OXE_GOOGLE` | `oxe_google` | (OXE) | (OXE) | (OXE) | varies |
| `OXE_WIDOWX` | `oxe_widowx` | (OXE) | (OXE) | (OXE) | varies |
| `OXE_DROID` | `oxe_droid` | (OXE) | (OXE) | (OXE) | varies |
| `NEW_EMBODIMENT` | `new_embodiment` | ❌ NOT in processor configs — KeyError! | - | - | - |

**Important:** `NEW_EMBODIMENT` tag is **not supported** by N1.6's processor. It raises `KeyError: 'new_embodiment'` in `processor.get_modality_configs()[embodiment_tag.value]`. For fine-tuned models, the tag must match one of the known embodiments, or users must register a custom embodiment config.

---

## Testing Strategy

### Unit Tests
```python
class TestObservationMapping:
    def test_explicit_mapping_no_positional_remap(self):
        """Explicit mapping should produce exact model keys, not positional."""
        mapping = ObservationMapping(
            video={"my_cam": "ego_view_bg_crop_pad_res256_freq20"},
            state={"my_joints": ("left_arm", 7), "my_grip": ("left_hand", 6)},
            language_key="task",
        )
        # ... verify output keys match model exactly

    def test_auto_infer_exact_match(self):
        """When our config keys match model keys, auto-infer should work."""
        
    def test_auto_infer_positional_fallback_warns(self):
        """Positional fallback should log a warning."""
        
    def test_missing_robot_key_zero_fills(self):
        """If robot doesn't provide a mapped key, zero-fill with correct DOF."""
        
    def test_extra_robot_keys_ignored(self):
        """Robot keys not in mapping are silently dropped."""
        
    def test_dof_from_model_normalizer(self):
        """DOF should be discovered from model, not guessed."""

class TestActionMapping:
    def test_explicit_reverse_mapping(self):
        """Model action keys map back to robot actuator names."""
        
    def test_unmapped_action_keys_prefixed(self):
        """Model actions without mapping get 'unmapped.' prefix."""
```

### Integration Tests
```python
def test_explicit_mapping_so100_on_gr1(local_policy):
    """SO-100 with explicit GR1 mapping should produce valid actions."""
    policy = Gr00tPolicy(
        model_path="nvidia/GR00T-N1.6-3B",
        embodiment_tag="GR1",
        observation_mapping={
            "webcam": "video.ego_view_bg_crop_pad_res256_freq20",
            "joint_position": "state.left_arm",
            "gripper": "state.left_hand",
        },
        action_mapping={
            "action.left_arm": "joint_position",
            "action.left_hand": "gripper",
        },
    )
    
    robot_obs = {
        "webcam": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        "joint_position": np.random.uniform(-1, 1, 5).astype(np.float32),
        "gripper": np.array([0.5], dtype=np.float32),
    }
    
    actions = await policy.get_actions(robot_obs, instruction="pick up cube")
    assert all("joint_position" in a for a in actions)
    assert all("gripper" in a for a in actions)
```

---

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Key namespaces | 3 (robot, data config, model) | 2 (robot, model) — data config becomes optional sugar |
| Mapping type | Implicit positional | Explicit declarative |
| Failure mode | Silent wrong mapping | KeyError at construction or clear warning |
| Zero-fill DOF | Guessed from first state key | Read from model config |
| Adding new robot | Edit `_data_config_defs.py` + `_STATE_LAYOUTS` + pray order is right | Write one YAML file or pass one dict |
| Code complexity | ~150 lines of mapping + remapping | ~100 lines of direct transform |
| Debugging | "Why is left_arm getting gripper values?" | `logger.info("Mapped webcam → video.ego_view_bg_crop_pad_res256_freq20")` |

**The mapping should be data, not code.**
