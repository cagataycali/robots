"""Real-pipeline validation for the LeRobot ProcessorBridge.

These tests exercise :class:`ProcessorBridge` against an ACTUAL LeRobot
``DataProcessorPipeline`` built directly from processor steps (no network, no
model download), so the bridge wiring is verified against real LeRobot internals
rather than mocks. The mock-heavy suite in ``test_policy.py`` covers error paths;
this file proves the success path against the installed LeRobot version.

Skips cleanly when LeRobot's processor framework is unavailable.
"""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("lerobot.processor.pipeline")

from lerobot.processor import IdentityProcessorStep  # noqa: E402
from lerobot.processor.pipeline import (  # noqa: E402
    ComplementaryDataProcessorStep,
    DataProcessorPipeline,
)
from lerobot.processor.rename_processor import RenameObservationsProcessorStep  # noqa: E402

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap  # noqa: E402
from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402


def _real_preprocessor(rename_map: dict[str, str] | None = None) -> DataProcessorPipeline:
    """Build a real (network-free) preprocessor pipeline with a rename step."""
    return DataProcessorPipeline(steps=[RenameObservationsProcessorStep(rename_map=dict(rename_map or {}))])


def _real_postprocessor() -> DataProcessorPipeline:
    """Build a real (network-free) postprocessor pipeline (identity action step)."""
    return DataProcessorPipeline(steps=[IdentityProcessorStep()])


class TestApplyEmbodimentRealPipeline:
    """apply_embodiment against a real DataProcessorPipeline."""

    def test_mutates_existing_rename_step_and_inserts_pack_state(self):
        pipe = _real_preprocessor()
        bridge = ProcessorBridge(preprocessor=pipe, device="cpu")
        em = EmbodimentMap(
            name="t",
            obs_rename={"front": "observation.images.top", "wrist": "observation.images.wrist"},
            state_keys=["1", "2", "3", "4", "5", "6"],
            action_keys=["1", "2", "3", "4", "5", "6"],
            dim_policy="pad",
        )
        bridge.apply_embodiment(em, input_features={})

        names = [getattr(s, "_registry_name", type(s).__name__) for s in pipe.steps]
        assert names[0] == "rename_observations_processor"
        assert names[1] == "strands_pack_state"
        # rename_map was set on the EXISTING step in place (not a new one prepended).
        assert pipe.steps[0].rename_map == em.obs_rename

    def test_inserts_rename_step_when_pipeline_has_none(self):
        # Pipeline with no rename step at all -> bridge must insert one at front.
        pipe = DataProcessorPipeline(steps=[])
        bridge = ProcessorBridge(preprocessor=pipe)
        em = EmbodimentMap(
            name="t",
            obs_rename={"front": "observation.images.top"},
            state_keys=["1", "2"],
            dim_policy="pad",
        )
        bridge.apply_embodiment(em, input_features=None)

        names = [getattr(s, "_registry_name", type(s).__name__) for s in pipe.steps]
        assert names[0] == "rename_observations_processor"
        assert names[1] == "strands_pack_state"
        assert pipe.steps[0].rename_map == em.obs_rename

    def test_idempotent_reapply_keeps_single_pack_state(self):
        pipe = _real_preprocessor()
        bridge = ProcessorBridge(preprocessor=pipe)
        em = EmbodimentMap(name="t", obs_rename={}, state_keys=["1", "2"], dim_policy="pad")
        bridge.apply_embodiment(em, input_features={})
        bridge.apply_embodiment(em, input_features={})

        names = [getattr(s, "_registry_name", type(s).__name__) for s in pipe.steps]
        assert names.count("strands_pack_state") == 1

    def test_no_preprocessor_is_noop(self):
        bridge = ProcessorBridge()  # passthrough, no pipelines
        em = EmbodimentMap(name="t", obs_rename={"a": "b"}, state_keys=["1"], dim_policy="pad")
        # Must not raise; simply does nothing.
        bridge.apply_embodiment(em, input_features={})
        assert not bridge.has_preprocessor

    def test_no_state_keys_skips_pack_state(self):
        pipe = _real_preprocessor()
        bridge = ProcessorBridge(preprocessor=pipe)
        em = EmbodimentMap(
            name="t",
            obs_rename={"front": "observation.images.top"},
            state_keys=[],
            dim_policy="pad",
        )
        bridge.apply_embodiment(em, input_features={})
        names = [getattr(s, "_registry_name", type(s).__name__) for s in pipe.steps]
        assert "strands_pack_state" not in names

    def test_expected_dim_from_input_features_pads_state(self):
        # Model declares an 8-dim observation.state; embodiment provides 6 joints
        # with dim_policy="pad" -> packed state should be padded to 8.
        pipe = _real_preprocessor({})
        bridge = ProcessorBridge(preprocessor=pipe)
        em = EmbodimentMap(
            name="t",
            obs_rename={},
            state_keys=["1", "2", "3", "4", "5", "6"],
            dim_policy="pad",
        )
        input_features = {"observation.state": type("F", (), {"shape": (8,)})()}
        bridge.apply_embodiment(em, input_features=input_features)

        raw = {str(i): float(i) / 10 for i in range(1, 7)}
        out = bridge.preprocess(raw)
        state = np.asarray(out["observation.state"]).ravel().tolist()
        assert len(state) == 8
        assert state[:6] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert state[6:] == [0.0, 0.0]


class TestPreprocessRealPipeline:
    """preprocess against a real DataProcessorPipeline."""

    def test_rename_and_pack_with_instruction(self):
        pipe = _real_preprocessor()
        bridge = ProcessorBridge(preprocessor=pipe, device="cpu")
        em = EmbodimentMap(
            name="t",
            obs_rename={"front": "observation.images.top"},
            state_keys=["1", "2", "3"],
            dim_policy="pad",
        )
        bridge.apply_embodiment(em, input_features={})

        raw = {"front": np.zeros((4, 4, 3), dtype=np.uint8), "1": 0.1, "2": 0.2, "3": 0.3}
        out = bridge.preprocess(raw, instruction="pick up the cube")

        assert "observation.images.top" in out
        assert "observation.state" in out
        assert np.asarray(out["observation.state"]).ravel().tolist() == pytest.approx([0.1, 0.2, 0.3])
        # instruction surfaced as the LeRobot 'task' key
        assert out["task"] == "pick up the cube"
        # raw strands keys are consumed
        assert "front" not in out and "1" not in out

    def test_no_preprocessor_returns_observation_unchanged(self):
        bridge = ProcessorBridge()
        raw = {"front": np.zeros((2, 2, 3), dtype=np.uint8), "1": 0.5}
        out = bridge.preprocess(raw)
        assert out is raw

    def test_vla_complementary_data_merged_into_batch(self):
        """B10 regression: VLA steps that emit model-ready tensors into
        COMPLEMENTARY_DATA must be merged into the returned flat batch, else the
        policy's _model_inputs reads an empty dict (StopIteration).

        Simulates MolmoAct2's pack_inputs step with a real
        ComplementaryDataProcessorStep that injects model-ready keys.
        """

        class _PackInputsStep(ComplementaryDataProcessorStep):
            def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
                out = dict(complementary_data)
                # Model-ready tensors a VLA packer would emit:
                out["input_ids"] = [1, 2, 3]
                out["pixel_values"] = np.zeros((1, 3, 8, 8), dtype=np.float32)
                return out

            def get_config(self) -> dict[str, Any]:
                return {}

            def state_dict(self) -> dict[str, Any]:
                return {}

            def load_state_dict(self, state: dict[str, Any]) -> None:
                pass

            def transform_features(self, features):
                return features

        pipe = DataProcessorPipeline(steps=[RenameObservationsProcessorStep(rename_map={}), _PackInputsStep()])
        bridge = ProcessorBridge(preprocessor=pipe)

        raw = {"observation.state": [0.0, 0.0]}
        out = bridge.preprocess(raw, instruction="grasp")

        # complementary keys (packed inputs + task) merged into the flat batch
        assert "input_ids" in out
        assert "pixel_values" in out
        assert out["task"] == "grasp"
        # observation keys still present
        assert "observation.state" in out

    def test_observation_keys_win_over_complementary_on_conflict(self):
        """When a key exists in both OBSERVATION and COMPLEMENTARY_DATA, the
        canonical normalized observation value must win."""

        class _ConflictStep(ComplementaryDataProcessorStep):
            def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
                out = dict(complementary_data)
                out["observation.state"] = "FROM_COMPLEMENTARY"
                return out

            def get_config(self) -> dict[str, Any]:
                return {}

            def state_dict(self) -> dict[str, Any]:
                return {}

            def load_state_dict(self, state: dict[str, Any]) -> None:
                pass

            def transform_features(self, features):
                return features

        pipe = DataProcessorPipeline(steps=[RenameObservationsProcessorStep(rename_map={}), _ConflictStep()])
        bridge = ProcessorBridge(preprocessor=pipe)
        raw = {"observation.state": [1.0, 2.0]}
        out = bridge.preprocess(raw, instruction="x")
        assert out["observation.state"] == [1.0, 2.0]


class TestPostprocessRealPipeline:
    """postprocess against a real DataProcessorPipeline."""

    def test_identity_action_passthrough(self):
        import torch

        bridge = ProcessorBridge(postprocessor=_real_postprocessor())
        action = torch.tensor([1.0, 2.0, 3.0])
        out = bridge.postprocess(action)
        assert np.asarray(out).ravel().tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_no_postprocessor_returns_action_unchanged(self):
        bridge = ProcessorBridge()
        sentinel = object()
        assert bridge.postprocess(sentinel) is sentinel


class TestBridgeStateRealPipeline:
    """Diagnostic surface against real pipelines."""

    def test_get_info_and_repr_reflect_loaded_pipelines(self):
        bridge = ProcessorBridge(preprocessor=_real_preprocessor(), postprocessor=_real_postprocessor())
        info = bridge.get_info()
        assert info["has_preprocessor"] is True
        assert info["has_postprocessor"] is True
        assert info["is_active"] is True
        assert "pre=1steps" in info["repr"]
        assert "post=1steps" in info["repr"]

    def test_reset_propagates_to_both_pipelines(self):
        # reset() on real pipelines must not raise.
        bridge = ProcessorBridge(preprocessor=_real_preprocessor(), postprocessor=_real_postprocessor())
        bridge.reset()
        assert bridge.is_active
