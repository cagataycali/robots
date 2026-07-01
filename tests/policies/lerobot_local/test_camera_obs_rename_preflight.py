"""Camera <-> embodiment ``obs_rename`` alignment is validated before download.

A VLA checkpoint (e.g. MolmoAct2 SO-100/101) declares image input features that
the embodiment's ``obs_rename`` feeds from named runtime keys
(``front`` -> ``observation.images.image``). When the sim's attached camera
names do not match any source key for a model image feature, the mismatch used
to surface only deep inside the preprocessor AFTER a multi-minute weight
download, as a confusing "image_keys missing from observation" failure.

These tests pin the fail-fast behavior:

* :meth:`LerobotLocalPolicy.preflight` raises a ``ValueError`` naming the
  expected camera source keys when none are present, passes when they are, and
  is satisfied by an ``obs_rename_override`` that maps a present camera onto the
  model feature;
* :func:`strands_robots.policies.preflight_policy` dispatches to the resolved
  provider class without instantiating it (a no-op for providers that do not
  override ``preflight``);
* ``SimEngine.run_policy`` / ``eval_policy`` short-circuit with a
  ``status=error`` result BEFORE ``create_policy`` is ever called;
* ``ProcessorBridge`` enriches the deep "image_keys missing" failure with the
  expected camera source keys and what the observation actually provided;
* ``obs_rename_override`` merges over the embodiment's ``obs_rename`` when the
  policy configures its embodiment.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.policies import preflight_policy
from strands_robots.policies.base import Policy
from strands_robots.policies.lerobot_local import processor as processor_mod
from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import ProcessorBridge
from strands_robots.simulation.base import SimEngine

SO101_CAMS = {"front", "wrist"}
SO101_JOINTS = {"1", "2", "3", "4", "5", "6"}


# ---------------------------------------------------------------------------
# LerobotLocalPolicy.preflight (the embodiment-aware check)
# ---------------------------------------------------------------------------
def test_preflight_passes_when_camera_names_match_embodiment():
    """No raise when the observation carries the embodiment's source keys."""
    LerobotLocalPolicy.preflight(SO101_CAMS | SO101_JOINTS, embodiment="so101")


def test_preflight_raises_naming_expected_cameras_on_mismatch():
    """Mismatched camera names raise, naming both the model features and fix."""
    with pytest.raises(ValueError) as ei:
        LerobotLocalPolicy.preflight({"realsense_top", "realsense_side"} | SO101_JOINTS, embodiment="so101")
    msg = str(ei.value)
    # Expected camera source names the user must provide.
    assert "front" in msg and "wrist" in msg
    # Names the model image features it could not satisfy.
    assert "observation.images.image" in msg
    # Points at both escape hatches.
    assert "obs_rename_override" in msg
    # Shows what the runtime actually provided.
    assert "realsense_top" in msg


def test_preflight_satisfied_by_obs_rename_override():
    """An override mapping a present camera onto the feature satisfies preflight."""
    LerobotLocalPolicy.preflight(
        {"realsense_top", "realsense_side"} | SO101_JOINTS,
        embodiment="so101",
        obs_rename_override={
            "realsense_top": "observation.images.image",
            "realsense_side": "observation.images.wrist_image",
        },
    )


def test_preflight_partial_override_still_raises_for_unmapped_feature():
    """Overriding only one feature still raises for the other unmapped one."""
    with pytest.raises(ValueError) as ei:
        LerobotLocalPolicy.preflight(
            {"realsense_top"} | SO101_JOINTS,
            embodiment="so101",
            obs_rename_override={"realsense_top": "observation.images.image"},
        )
    # The wrist feature remains unsatisfied; its default source key is named.
    assert "wrist" in str(ei.value)
    assert "observation.images.wrist_image" in str(ei.value)


def test_preflight_noop_without_embodiment():
    """No embodiment -> the legacy heuristic path; preflight cannot reason, no-op."""
    LerobotLocalPolicy.preflight({"whatever"})  # must not raise


def test_preflight_noop_for_unknown_embodiment_name():
    """Unknown embodiment names are left for create_policy to report."""
    LerobotLocalPolicy.preflight({"front"}, embodiment="totally_unknown_embodiment")


# ---------------------------------------------------------------------------
# Factory dispatch: preflight_policy
# ---------------------------------------------------------------------------
def test_preflight_policy_noop_for_provider_without_override():
    """A provider that does not override Policy.preflight is a no-op."""
    preflight_policy("mock", {"anything", "at", "all"})  # must not raise


def test_preflight_policy_dispatches_to_lerobot_local():
    """preflight_policy routes the check to the resolved lerobot_local class."""
    with pytest.raises(ValueError):
        preflight_policy("lerobot_local", {"realsense_top"}, embodiment="so101")


def test_preflight_policy_swallows_unresolvable_provider():
    """Resolution failures are not this hook's concern (create_policy reports)."""
    preflight_policy("definitely_not_a_provider", {"front"})  # must not raise


def test_base_policy_preflight_is_a_noop():
    """The ABC default is a no-op so most providers need no override."""
    assert Policy.preflight(set(), foo="bar") is None


# ---------------------------------------------------------------------------
# SimEngine.run_policy / eval_policy fail fast BEFORE create_policy
# ---------------------------------------------------------------------------
class _CamSim(SimEngine):
    """Minimal SimEngine stub whose observation carries named camera keys."""

    def __init__(self, camera_keys: tuple[str, ...]) -> None:
        self._joints = ["1", "2", "3", "4", "5", "6"]
        self._cams = list(camera_keys)

    def create_world(self, timestep=None, gravity=None, ground_plane=True):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def destroy(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def reset(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def step(self, n_steps: int = 1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_state(self):  # type: ignore[no-untyped-def]
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_robot(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return ["so101"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._joints)

    def add_object(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_object(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):  # type: ignore[no-untyped-def]
        obs: dict[str, object] = {j: 0.0 for j in self._joints}
        for c in self._cams:
            obs[c] = object()  # stand-in for an RGB frame; only keys matter
        return obs

    def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def render(self, camera_name="default", width=None, height=None):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}


def _exploding_create_policy(*_a, **_k):
    raise AssertionError("create_policy must not be called when preflight fails")


def test_run_policy_fails_fast_before_create_policy_on_camera_mismatch():
    """run_policy returns status=error from preflight, never reaching create_policy."""
    sim = _CamSim(camera_keys=("realsense_top", "realsense_side"))
    with patch("strands_robots.policies.create_policy", _exploding_create_policy):
        result = sim.run_policy(
            robot_name="so101",
            policy_provider="lerobot_local",
            policy_config={"embodiment": "so101", "pretrained_name_or_path": "x"},
            n_steps=2,
            control_frequency=10.0,
        )
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "front" in text and "wrist" in text


def test_eval_policy_fails_fast_before_create_policy_on_camera_mismatch():
    """eval_policy mirrors run_policy's fail-fast preflight."""
    sim = _CamSim(camera_keys=("realsense_top", "realsense_side"))
    with patch("strands_robots.policies.create_policy", _exploding_create_policy):
        result = sim.eval_policy(
            robot_name="so101",
            policy_provider="lerobot_local",
            policy_config={"embodiment": "so101", "pretrained_name_or_path": "x"},
            n_episodes=1,
            max_steps=2,
        )
    assert result["status"] == "error"
    assert "front" in result["content"][0]["text"]


def test_preflight_policy_config_returns_none_when_cameras_match():
    """Correct camera names let the preflight helper pass through (returns None)."""
    sim = _CamSim(camera_keys=("front", "wrist"))
    assert sim._preflight_policy_config("so101", "lerobot_local", {"embodiment": "so101"}) is None


# ---------------------------------------------------------------------------
# ProcessorBridge crash-site error enrichment
# ---------------------------------------------------------------------------
def test_camera_hint_enriches_image_keys_missing_error():
    """The deep 'image_keys missing' failure gains expected/actual camera names."""
    bridge = ProcessorBridge()
    bridge._obs_rename = {
        "front": "observation.images.image",
        "wrist": "observation.images.wrist_image",
    }
    hint = bridge._camera_hint(
        RuntimeError("MolmoAct2 image_keys missing from observation: [...]"),
        {"realsense_top": object(), "1": 0.0},
    )
    assert "front" in hint and "wrist" in hint
    assert "realsense_top" in hint
    assert "obs_rename_override" in hint


def test_camera_hint_empty_for_unrelated_error():
    """Unrelated pipeline failures get no camera hint."""
    bridge = ProcessorBridge()
    bridge._obs_rename = {"front": "observation.images.image"}
    assert bridge._camera_hint(RuntimeError("CUDA out of memory"), {"front": object()}) == ""


def test_camera_hint_empty_without_obs_rename():
    """No latched obs_rename (legacy path) -> no hint even on image-keys error."""
    bridge = ProcessorBridge()
    assert bridge._camera_hint(RuntimeError("image_keys missing"), {}) == ""


def test_camera_hint_empty_when_obs_rename_maps_no_image_features():
    """A state-only embodiment must not emit a camera hint on an image-keys error.

    ``_camera_hint`` names the *camera* source keys to fix, derived from the
    obs_rename entries whose target is an ``observation.images.*`` feature. When
    the latched obs_rename maps only proprioceptive state (no image targets),
    there are no camera keys to suggest, so the enrichment must stay empty rather
    than emit a hint listing zero expected keys. Regression pin for the
    ``if not expected`` guard: dropping it would surface a truncated
    "Expected camera source key(s): []" hint on non-vision policies.
    """
    bridge = ProcessorBridge()
    bridge._obs_rename = {"shoulder.pos": "observation.state"}
    assert bridge._camera_hint(RuntimeError("image_keys missing"), {"front": object()}) == ""


# ---------------------------------------------------------------------------
# Optional-dependency degradation of the module-level helpers
# ---------------------------------------------------------------------------
def test_try_import_processor_none_when_pipeline_class_absent(monkeypatch):
    """A lerobot build whose processor module lacks DataProcessorPipeline -> None.

    The bridge treats a missing pipeline class the same as a missing lerobot:
    it degrades to pass-through rather than raising, so a partial/older install
    does not crash policy construction.
    """
    monkeypatch.setattr(processor_mod, "require_optional", lambda *a, **k: object())
    assert processor_mod._try_import_processor() is None


def test_try_import_processor_none_when_lerobot_missing(monkeypatch):
    """A missing [lerobot] extra degrades to pass-through (None), never ImportError."""

    def _raise(*a, **k):
        raise ImportError("lerobot processor extra not installed")

    monkeypatch.setattr(processor_mod, "require_optional", _raise)
    assert processor_mod._try_import_processor() is None


def test_register_policy_processor_steps_noop_without_type():
    """No policy_type -> silent no-op (nothing imported, no crash)."""
    assert processor_mod._register_policy_processor_steps(None) is None


def test_register_policy_processor_steps_best_effort_on_import_failure(monkeypatch):
    """Every candidate module failing to import is swallowed; the helper returns.

    Step registration is best-effort: a failed import is logged at DEBUG and the
    caller proceeds (the pipeline load then raises its own clear error). A raise
    here would turn an optional-dep gap into a hard construction failure.
    """
    import importlib

    def _boom(name):
        raise ImportError(f"cannot import {name}")

    monkeypatch.setattr(importlib, "import_module", _boom)
    assert processor_mod._register_policy_processor_steps("act") is None


# ---------------------------------------------------------------------------
# obs_rename_override merges into the configured embodiment
# ---------------------------------------------------------------------------
def _feature(dim: int) -> MagicMock:
    feat = MagicMock()
    feat.shape = (dim,)
    return feat


def test_obs_rename_override_merges_into_configured_embodiment():
    """_configure_embodiment merges obs_rename_override over the embodiment map."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(
            embodiment=EmbodimentMap(
                name="so101_test",
                obs_rename={"front": "observation.images.image"},
                state_keys=["1", "2", "3", "4", "5", "6"],
                action_keys=["1", "2", "3", "4", "5", "6"],
                dim_policy="pad",
            ),
            obs_rename_override={"realsense_side": "observation.images.wrist_image"},
        )
    policy._input_features = {
        "observation.images.image": _feature(3),
        "observation.images.wrist_image": _feature(3),
        "observation.state": _feature(6),
    }
    policy._output_features = {"action": _feature(6)}
    bridge = MagicMock(name="ProcessorBridge")
    policy._processor_bridge = bridge

    policy._configure_embodiment()

    applied = bridge.apply_embodiment.call_args.args[0]
    assert applied.obs_rename == {
        "front": "observation.images.image",
        "realsense_side": "observation.images.wrist_image",
    }
