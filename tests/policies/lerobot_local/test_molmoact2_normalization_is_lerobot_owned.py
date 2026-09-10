"""MolmoAct2 normalization is read and refused by lerobot's own factory.

A transformers-native MolmoAct2 checkpoint carries its normalization statistics
in ``norm_stats.json``, keyed by embodiment tag. ``LerobotLocalPolicy`` routes
such a checkpoint to :func:`~strands_robots.policies.lerobot_local.molmoact2.build_policy`,
which builds the pre/post pipelines through lerobot's public factory
(``make_policy_config`` -> ``make_pre_post_processors``). lerobot's
``make_molmoact2_pre_post_processors`` reads ``norm_stats.json`` itself and
selects the requested tag, so strands neither parses that file nor reimplements
the numeric transform.

These cells pin that ownership from the outside, using only a checkpoint
directory the test writes: an undeclared tag is refused by lerobot with the tags
the file actually declares, a declared tag is accepted by that same reader, and
the processor bridge carries no norm-stats surface of its own.
"""

from __future__ import annotations

import inspect
import json
import traceback
from pathlib import Path

import pytest

from strands_robots.policies.lerobot_local import molmoact2
from strands_robots.policies.lerobot_local.processor import ProcessorBridge

TAG = "so100_so101_molmoact2"


def _require_molmoact2_runtime() -> None:
    """Skip unless lerobot's molmoact2 factory path is importable.

    ``build_policy`` needs the typed-feature API plus the auxiliary MolmoAct2
    runtime deps (transformers/peft/scipy). Gate on the symbols the code path
    actually uses so an older or partial environment skips rather than errors.
    """
    pytest.importorskip("lerobot")
    configs = pytest.importorskip("lerobot.configs")
    if not hasattr(configs, "FeatureType") or not hasattr(configs, "PolicyFeature"):
        pytest.skip("lerobot.configs lacks FeatureType/PolicyFeature (needs lerobot >= 0.5.2)")
    for dep in ("torch", "transformers", "peft", "scipy"):
        pytest.importorskip(dep)
    pytest.importorskip("lerobot.policies.molmoact2.processor_molmoact2")


def _checkpoint(tmp_path: Path) -> Path:
    """A transformers-native MolmoAct2 checkpoint dir carrying one tag's stats."""
    # Mirrors a real MolmoAct2 checkpoint's per-tag shape: quantiles, min/max,
    # and the gripper mask lerobot uses to decide which dimension it may skip.
    stats = {
        # The masked-out gripper dimension must already lie in [-1, 1]; lerobot
        # checks those values rather than trusting the mask.
        "q01": [-40.0] * 5 + [-1.0],
        "q99": [180.0] * 5 + [1.0],
        "min": [-120.0] * 5 + [-1.0],
        "max": [200.0] * 5 + [1.0],
        "mask": [True] * 5 + [False],
    }
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "molmoact2"}))
    (tmp_path / "norm_stats.json").write_text(
        json.dumps(
            {
                "norm_mode": "min_max",
                "metadata_by_tag": {TAG: {"action_stats": stats, "state_stats": stats}},
            }
        )
    )
    return tmp_path


def _build(ckpt: Path, norm_tag: str):
    """Run the molmoact2 load path, skipping only the checkpoint weight load."""
    return molmoact2.build_policy(
        str(ckpt),
        device="cpu",
        norm_tag=norm_tag,
        inference_action_mode="continuous",
        image_keys=["observation.images.top"],
        embodiment_spec=None,
        prebuilt_policy=object(),
    )


class TestLerobotOwnsMolmoAct2Normalization:
    def test_the_checkpoint_is_routed_to_the_molmoact2_path(self, tmp_path):
        """model_type=molmoact2 selects the wrapper, so lerobot's factory builds
        the pipelines (the generic ProcessorBridge load never runs)."""
        assert molmoact2.is_molmoact2(str(_checkpoint(tmp_path)), None) is True

    def test_an_undeclared_norm_tag_is_refused_by_lerobot(self, tmp_path):
        """lerobot names the tags the checkpoint's own file declares.

        The refusal quoting ``TAG`` is what proves lerobot read this file: the
        available-tag list can only come from parsing it.
        """
        _require_molmoact2_runtime()
        with pytest.raises(ValueError) as excinfo:
            _build(_checkpoint(tmp_path), "so999_typo")

        message = str(excinfo.value)
        assert "so999_typo" in message
        assert TAG in message
        raising_module = traceback.extract_tb(excinfo.tb)[-1].filename
        assert "lerobot/policies/molmoact2" in raising_module, raising_module

    def test_a_declared_norm_tag_is_accepted_by_that_reader(self, tmp_path):
        """A tag the file declares gets past normalization.

        The build still stops later, on the tokenizer/image-processor assets a
        real checkpoint ships (``processor_config.json``) - which is precisely
        the evidence that the norm-stats stage accepted the tag rather than
        refusing it.
        """
        _require_molmoact2_runtime()
        with pytest.raises((FileNotFoundError, ValueError)) as excinfo:
            _build(_checkpoint(tmp_path), TAG)

        message = str(excinfo.value)
        assert "Unknown MolmoAct2 norm_tag" not in message, f"the declared tag was refused: {message}"
        assert "processor_config.json" in message, message


class TestTheBridgeHasNoNormStatsSurface:
    """The generic bridge neither parses norm_stats.json nor takes a tag.

    Normalization for MolmoAct2 is lerobot's, so a second implementation here
    could drift from it silently. These pins keep that surface from returning.
    """

    def test_from_pretrained_takes_no_norm_tag(self):
        assert "norm_tag" not in inspect.signature(ProcessorBridge.from_pretrained).parameters

    def test_the_bridge_carries_no_norm_stats_loader(self):
        assert not hasattr(ProcessorBridge, "_load_norm_stats_fallback")
        assert "norm_stats" not in inspect.getsource(ProcessorBridge)

    def test_no_norm_stats_module_ships(self):
        with pytest.raises(ImportError):
            __import__("strands_robots.policies.lerobot_local.norm_stats")
