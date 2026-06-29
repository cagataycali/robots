"""Regression tests: reconcile a device-pinned processor step onto the host.

A LeRobot checkpoint trained on GPU bakes ``device_processor.device = "cuda"``
into its ``policy_preprocessor.json`` / ``policy_postprocessor.json``. Loaded on
a host without that device, LeRobot's ``get_safe_torch_device`` asserts the
device is available, so the ``device_processor`` step fails to instantiate and
``DataProcessorPipeline.from_pretrained`` raises a ``ValueError`` that is
indistinguishable from "no config file present".

Pre-fix, :meth:`ProcessorBridge.from_pretrained` swallowed that ValueError as a
missing config and returned a passthrough bridge -- normalization was silently
dropped, so observations reached the model un-normalized and predicted actions
reached the motors un-unnormalized (the arm barely moves). These tests pin the
behavior: the bridge knows its resolved target device and reconciles the pinned
step onto it instead of dropping the pipeline.

The tests build real (network-free) pipeline configs on disk and load them
through the real LeRobot pipeline, so they verify behavior against actual
LeRobot internals rather than mocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("lerobot.processor.pipeline")

from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402


def _unavailable_device() -> str:
    """Return a real accelerator device string that is NOT available here.

    Reproducing the bug needs a ``device_processor`` pinned to a device whose
    ``get_safe_torch_device`` assertion fails on this host. ``"cuda"`` covers
    CPU-only CI and CPU-only edge hosts; ``"mps"`` covers CUDA GPU hosts. Skip
    only on the rare host where both are available.
    """
    import torch

    if not torch.cuda.is_available():
        return "cuda"
    if not torch.backends.mps.is_available():
        return "mps"
    # Both accelerators present: no unavailable device to pin. Return the
    # skip (typed NoReturn) so every exit path is an explicit return and the
    # function has no implicit fall-through.
    return pytest.skip("host has both CUDA and MPS; cannot pin an unavailable device")


def _write_pipeline_config(directory: Path, filename: str, device: str) -> None:
    """Write a minimal one-step (device_processor) pipeline config to disk."""
    config = {
        "name": filename.removesuffix(".json"),
        "steps": [{"registry_name": "device_processor", "config": {"device": device, "float_dtype": None}}],
    }
    (directory / filename).write_text(json.dumps(config))


def test_device_pinned_preprocessor_reconciled_to_target_device(tmp_path: Path) -> None:
    """A device-pinned preprocessor loads (not dropped) by reconciling the device.

    Fails pre-fix: the unavailable-device assertion is swallowed as "no config"
    and the bridge is an inert passthrough (``has_preprocessor`` False).
    """
    pinned = _unavailable_device()
    _write_pipeline_config(tmp_path, "policy_preprocessor.json", pinned)

    bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")

    assert bridge.has_preprocessor is True
    assert bridge.is_active is True


def test_device_pinned_postprocessor_reconciled_to_target_device(tmp_path: Path) -> None:
    """The same reconciliation covers the postprocessor (un-normalization) path."""
    pinned = _unavailable_device()
    _write_pipeline_config(tmp_path, "policy_postprocessor.json", pinned)

    bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")

    assert bridge.has_postprocessor is True
    assert bridge.is_active is True


def test_user_device_processor_override_is_not_clobbered(tmp_path: Path) -> None:
    """An explicit user device_processor override is honored as-is (no retry)."""
    pinned = _unavailable_device()
    _write_pipeline_config(tmp_path, "policy_preprocessor.json", pinned)

    bridge = ProcessorBridge.from_pretrained(
        str(tmp_path), device="cpu", overrides={"device_processor": {"device": "cpu"}}
    )

    assert bridge.has_preprocessor is True


def test_genuinely_missing_config_stays_passthrough(tmp_path: Path) -> None:
    """No config on disk -> passthrough bridge, no spurious device retry."""
    bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")

    assert bridge.has_preprocessor is False
    assert bridge.has_postprocessor is False
    assert bridge.is_active is False


def test_unknown_device_cannot_reconcile_and_stays_passthrough(tmp_path: Path) -> None:
    """With no resolved target device the bridge cannot reconcile and stays inert.

    Documents the boundary: reconciliation requires knowing the host device.
    """
    pinned = _unavailable_device()
    _write_pipeline_config(tmp_path, "policy_preprocessor.json", pinned)

    bridge = ProcessorBridge.from_pretrained(str(tmp_path), device=None)

    assert bridge.has_preprocessor is False
