# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``wbc`` checkpoint given as a HuggingFace id resolves to its snapshot before the config reads it.

``WBCPolicy(checkpoint="<org>/<repo>")``, which the provider accepts, downloaded
the ONNX files and then failed ``WBCPolicy main ONNX checkpoint not found
(resolved: '<hf cache>/.../snapshots/<sha>/nepyope/GR00T-WholeBodyControl_g1')``,
while the same files as a local directory built and walked the G1 0.665 m in
2 s. ``_resolve_config`` ran first and read the id as a path: no directory of
that name, so ``_default_onnx_paths`` took the string for the main ONNX file
itself; ``_load_sessions`` downloaded the snapshot afterwards and resolved that
relative "file" against it. The download now happens in the constructor before
the config is resolved, so the config sees the snapshot directory the way it
sees a local checkout: its ``config.json`` when it ships one, its canonical ONNX
names otherwise. The hub is faked; nothing here reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import strands_robots.policies.wbc.policy as wbc_policy
from strands_robots.policies.wbc.policy import WBCPolicy
from tests.policies.wbc.test_policy import _make_config

_HUB_ID = "nepyope/GR00T-WholeBodyControl_g1"


class _FakeHub:
    """``huggingface_hub`` as ``_maybe_download_checkpoint`` uses it: one call, one directory."""

    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self.calls: list[str] = []

    def snapshot_download(self, repo_id: str, allow_patterns: Any = None) -> str:
        self.calls.append(repo_id)
        return str(self.snapshot)


@pytest.fixture
def canonical_snapshot(tmp_path: Path) -> Path:
    """The upstream layout: the two canonical ONNX names and no config.json."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "GR00T-WholeBodyControl-Balance.onnx").touch()
    (snapshot / "GR00T-WholeBodyControl-Walk.onnx").touch()
    return snapshot


def _build_with_hub(monkeypatch: pytest.MonkeyPatch, snapshot: Path, **kwargs: Any) -> tuple[WBCPolicy, _FakeHub, list]:
    """Construct against the fake hub with the session load recorded instead of run."""
    hub = _FakeHub(snapshot)
    monkeypatch.setattr(wbc_policy, "require_optional", lambda *a, **k: hub)
    loads: list[str | None] = []
    with patch.object(WBCPolicy, "_load_sessions", lambda self, checkpoint: loads.append(checkpoint)):
        policy = WBCPolicy(checkpoint=_HUB_ID, **kwargs)
    return policy, hub, loads


class TestTheIdIsResolvedBeforeTheConfig:
    def test_the_config_points_at_the_snapshots_canonical_files(
        self, monkeypatch: pytest.MonkeyPatch, canonical_snapshot: Path
    ) -> None:
        policy, hub, loads = _build_with_hub(monkeypatch, canonical_snapshot)
        assert hub.calls == [_HUB_ID]
        assert policy._config.policy_path == str(canonical_snapshot / "GR00T-WholeBodyControl-Balance.onnx")
        assert policy._config.walk_policy_path == str(canonical_snapshot / "GR00T-WholeBodyControl-Walk.onnx")

    def test_the_session_loader_receives_the_local_directory(
        self, monkeypatch: pytest.MonkeyPatch, canonical_snapshot: Path
    ) -> None:
        _policy, _hub, loads = _build_with_hub(monkeypatch, canonical_snapshot)
        assert loads == [str(canonical_snapshot)]

    def test_the_snapshots_config_json_is_honoured(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "policy.onnx").touch()
        shipped = _make_config(action_scale=0.125)
        (snapshot / "config.json").write_text(json.dumps(dataclasses.asdict(shipped)), encoding="utf-8")
        policy, _hub, _loads = _build_with_hub(monkeypatch, snapshot)
        assert policy._config.action_scale == 0.125

    def test_the_download_happens_once(self, monkeypatch: pytest.MonkeyPatch, canonical_snapshot: Path) -> None:
        """``_load_sessions`` resolves the checkpoint too; an existing directory is returned as-is."""
        hub = _FakeHub(canonical_snapshot)
        monkeypatch.setattr(wbc_policy, "require_optional", lambda *a, **k: hub)
        seen: list[str | None] = []
        real = WBCPolicy._load_sessions

        def loader(self: WBCPolicy, checkpoint: str | None) -> None:
            seen.append(checkpoint)
            # The real loader's first step, without onnxruntime: the resolver
            # must hand the directory back unchanged and not dial the hub again.
            assert WBCPolicy._maybe_download_checkpoint(checkpoint) == checkpoint

        with patch.object(WBCPolicy, "_load_sessions", loader):
            WBCPolicy(checkpoint=_HUB_ID)
        assert seen == [str(canonical_snapshot)]
        assert hub.calls == [_HUB_ID]
        assert real is not loader

    def test_an_explicit_config_still_gets_the_local_directory(
        self, monkeypatch: pytest.MonkeyPatch, canonical_snapshot: Path
    ) -> None:
        policy, hub, loads = _build_with_hub(monkeypatch, canonical_snapshot, config=_make_config())
        assert hub.calls == [_HUB_ID]
        assert loads == [str(canonical_snapshot)]
        assert policy._config.policy_path == "policy.onnx"


class TestWhatDoesNotChange:
    def test_a_local_directory_is_not_dialled(self, monkeypatch: pytest.MonkeyPatch, canonical_snapshot: Path) -> None:
        class _Boom:
            def snapshot_download(self, *a: Any, **k: Any) -> str:
                raise AssertionError("a local directory must not reach the hub")

        monkeypatch.setattr(wbc_policy, "require_optional", lambda *a, **k: _Boom())
        with patch.object(WBCPolicy, "_load_sessions", lambda self, checkpoint: None):
            policy = WBCPolicy(checkpoint=str(canonical_snapshot))
        assert policy._config.policy_path == str(canonical_snapshot / "GR00T-WholeBodyControl-Balance.onnx")

    def test_the_stub_seam_makes_no_network_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``allow_missing_models=True`` loads no session, so it downloads nothing either."""

        class _Boom:
            def snapshot_download(self, *a: Any, **k: Any) -> str:
                raise AssertionError("the stub seam must not reach the hub")

        monkeypatch.setattr(wbc_policy, "require_optional", lambda *a, **k: _Boom())
        policy = WBCPolicy(checkpoint=_HUB_ID, config=_make_config(), allow_missing_models=True)
        assert policy.policy_session is None

    def test_a_hub_failure_is_still_the_actionable_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Down:
            def snapshot_download(self, *a: Any, **k: Any) -> str:
                raise OSError("network down")

        monkeypatch.setattr(wbc_policy, "require_optional", lambda *a, **k: _Down())
        with pytest.raises(RuntimeError, match="failed to download checkpoint"):
            WBCPolicy(checkpoint=_HUB_ID)
