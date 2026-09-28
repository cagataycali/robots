"""The WBC weights route fetches the two controllers, not the tree around them.

The checkpoint paragraph on ``docs/learn/policies/wbc.md`` is the first thing a
WBC user acts on, and the two G1 controllers it needs are 1.8 MB each: obtaining
them by cloning ``NVlabs/GR00T-WholeBodyControl`` downloads a 4.6 GB git-LFS
tree for 3.6 MB of ONNX. The page no longer types a shell recipe; it tells the
reader ``checkpoint`` takes a HuggingFace model id and names the artifact files
the loader accepts. So the route graded is the loader's own download (it must
pull the ONNX and JSON artifacts, never the whole snapshot), the page must not
reintroduce a whole-repository clone, and the filenames it names are pinned to
the constants :class:`~strands_robots.policies.wbc.WBCPolicy` resolves (a
transcription can drift from the loader; a constant cannot).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies.wbc import policy as wbc_policy
from strands_robots.policies.wbc.policy import (
    _MAIN_POLICY_CANONICAL,
    _MAIN_POLICY_FILENAME,
    _WALK_POLICY_CANONICAL,
    _WALK_POLICY_FILENAME,
    WBCPolicy,
)

_PAGE = Path(__file__).resolve().parents[3] / "docs" / "learn" / "policies" / "wbc.md"
_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_ONNX = re.compile(r"[\w.-]+\.onnx")
_ARTIFACT_ONLY = {"*.onnx", "*.json"}
_ACCEPTED = {_MAIN_POLICY_CANONICAL, _WALK_POLICY_CANONICAL, _MAIN_POLICY_FILENAME, _WALK_POLICY_FILENAME}


def _page() -> str:
    return _PAGE.read_text(encoding="utf-8")


def _fences() -> list[str]:
    return [m.group(1) for m in _FENCE.finditer(_page())]


def _sentences_with(text: str, needle: str) -> list[str]:
    """Sentences of ``text`` (outside fences) that contain ``needle``."""
    prose = _FENCE.sub("", text)
    return [s for s in re.split(r"(?<=\.)\s+", prose) if needle in s]


def _paragraphs_with(text: str, needle: str) -> list[str]:
    """Paragraphs of ``text`` (outside fences) that contain ``needle``."""
    prose = _FENCE.sub("", text)
    return [par for par in re.split(r"\n\s*\n", prose) if needle in par]


def _expand_abbreviated(names: list[str]) -> set[str]:
    """``X-Balance.onnx`` and ``-Walk.onnx`` reads as two files sharing a prefix."""
    out: set[str] = set()
    prefix: str | None = None
    for name in names:
        if name.startswith("-") and prefix is not None:
            out.add(prefix + name)
        else:
            out.add(name)
            prefix = name.rsplit("-", 1)[0] if "-" in name else None
    return out


@pytest.fixture(scope="module")
def checkpoint_paragraph() -> str:
    """The prose that tells the reader what ``checkpoint`` accepts."""
    hits = _paragraphs_with(_page(), "HuggingFace model id")
    assert hits, f"{_PAGE.name} no longer says checkpoint takes a HuggingFace model id"
    return " ".join(hits)


def test_no_fence_fetches_the_weights_by_cloning_the_repository() -> None:
    """A whole-repo clone pulls the 4.6 GB LFS tree to obtain 3.6 MB of weights."""
    assert _fences(), f"premise: {_PAGE.name} has no fences"
    for fence in _fences():
        assert "git clone" not in fence, (
            f"{_PAGE.name} fetches the WBC weights by cloning the upstream repository; "
            "the two controllers are 1.8 MB each and the clone is a 4.6 GB git-LFS tree. "
            "Point checkpoint at a HuggingFace model id or fetch the two artifacts directly."
        )


def test_the_model_id_route_the_page_names_downloads_only_the_artifacts(
    checkpoint_paragraph: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loader's HuggingFace download is per-artifact, not the whole snapshot."""
    assert "model id" in checkpoint_paragraph
    calls: list[dict[str, Any]] = []

    class _Hub:
        @staticmethod
        def snapshot_download(**kwargs: Any) -> str:
            calls.append(kwargs)
            return str(tmp_path)

    monkeypatch.setattr(wbc_policy, "require_optional", lambda *a, **k: _Hub())
    resolved = WBCPolicy._maybe_download_checkpoint("nvidia/GR00T-WholeBodyControl")
    assert resolved == str(tmp_path)
    assert len(calls) == 1, "the model id route did not go through huggingface_hub"
    patterns = set(calls[0].get("allow_patterns") or ())
    assert patterns == _ARTIFACT_ONLY, (
        f"WBCPolicy downloads {sorted(patterns) or 'the whole snapshot'} for a model id; "
        "the page promises the artifacts, so the loader must restrict to ONNX and JSON"
    )


def test_the_page_names_both_controllers_the_loader_accepts(checkpoint_paragraph: str) -> None:
    """Both official artifact names, spelled as the loader resolves them."""
    accepts = [s for s in re.split(r"(?<=\.)\s+", checkpoint_paragraph) if "accepts" in s]
    assert accepts, f"{_PAGE.name} no longer states which artifact names the loader accepts"
    named = _expand_abbreviated(_ONNX.findall(" ".join(accepts)))
    assert {_MAIN_POLICY_CANONICAL, _WALK_POLICY_CANONICAL} <= named, (
        f"{_PAGE.name} must name both the main and the walk controller, got {sorted(named)}"
    )
    unknown = named - _ACCEPTED
    assert not unknown, f"{_PAGE.name} names ONNX files the loader does not resolve: {sorted(unknown)}"


def test_the_page_names_the_default_filenames_the_loader_looks_for() -> None:
    """The ``policy.onnx`` / ``walk_policy.onnx`` pair on the page is the loader's default pair."""
    named = set(_ONNX.findall(_FENCE.sub("", _page())))
    assert {_MAIN_POLICY_FILENAME, _WALK_POLICY_FILENAME} <= named, (
        f"{_PAGE.name} must name the default ONNX pair the loader resolves, got {sorted(named)}"
    )


def test_the_sonic_files_the_page_calls_refused_are_not_controller_names() -> None:
    """The GEAR-SONIC files the page says are refused must not be names the loader accepts."""
    refused = _sentences_with(_page(), "GEAR-SONIC")
    assert refused, f"{_PAGE.name} no longer warns about the GEAR-SONIC repository"
    named = set(_ONNX.findall(" ".join(refused)))
    assert named, "premise: the GEAR-SONIC sentence names no ONNX file"
    assert not (named & _ACCEPTED), f"{_PAGE.name} calls a controller name refused: {sorted(named & _ACCEPTED)}"


def test_the_abbreviation_reader_expands_the_pair_and_reports_a_planted_name() -> None:
    """The grader reads the page's ``-Walk.onnx`` shorthand and still sees a wrong name."""
    assert _expand_abbreviated(["GR00T-WholeBodyControl-Balance.onnx", "-Walk.onnx"]) == {
        _MAIN_POLICY_CANONICAL,
        _WALK_POLICY_CANONICAL,
    }
    assert "GR00T-WholeBodyControl-Run.onnx" in _expand_abbreviated(
        ["GR00T-WholeBodyControl-Balance.onnx", "-Run.onnx"]
    )
