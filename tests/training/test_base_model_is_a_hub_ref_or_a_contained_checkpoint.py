"""``TrainSpec.base_model`` is a Hub reference or a checkpoint inside a known home, nothing else.

The field is the warm start every trainer reads: lerobot's ``--policy.pretrained_path``,
the GR00T ``--base_model_path``, the Cosmos DCP conversion source. Preflight refused only a
leading dash. Any other string went through, so an agent could name ``/etc``, ``~/.ssh`` or
a path with ``..`` in it and learn from the backend's own error whether the directory exists
and what it holds (a read side path oracle), or point the warm start at a directory the
operator never meant a training run to read.

The rule is now the one the dashboard already applies to a client-named checkpoint: a Hub
id (``org/name``, ``name``, optionally ``@revision``) passes through, anything that looks like
a path must resolve inside the training output home or the Hub cache, and the refusal is one
sentence whether the target exists or not.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from strands_robots.training import _validate
from strands_robots.training.base import TrainSpec


@pytest.fixture
def homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    out = tmp_path / "out"
    hub = tmp_path / "hub"
    out.mkdir()
    hub.mkdir()
    monkeypatch.setenv("STRANDS_TRAIN_OUTPUT_DIR", str(out))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.delenv("HF_HOME", raising=False)
    return {"out": out.resolve(), "hub": hub.resolve()}


def _spec(base_model: str, tmp_path: Path) -> TrainSpec:
    return TrainSpec(
        dataset_root=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out" / "run"),
        base_model=base_model,
        steps=1,
    )


@pytest.mark.parametrize(
    "ref",
    ["lerobot/act_aloha_sim", "nvidia/GR00T-N1.5-3B", "m", "lerobot/smolvla_base@v2.1", "org/name@abc123"],
)
def test_a_hub_reference_passes(homes, tmp_path, ref):
    assert _validate.base_model_error(ref) is None
    assert not [p for p in _validate.validate_train_inputs(_spec(ref, tmp_path)) if "base_model" in p]


def test_a_checkpoint_inside_the_output_home_passes(homes, tmp_path):
    ckpt = homes["out"] / "run" / "checkpoints" / "last" / "pretrained_model"
    assert _validate.base_model_error(str(ckpt)) is None, "containment does not require the path to exist"


def test_a_snapshot_inside_the_hub_cache_passes(homes, tmp_path):
    snap = homes["hub"] / "models--lerobot--act" / "snapshots" / "abc"
    assert _validate.base_model_error(str(snap)) is None


@pytest.mark.parametrize(
    "outside",
    ["/etc", "~/.ssh", "/nonexistent/anything", "./relative/dir", "../up", "a/b/c", "C:\\Windows\\System32"],
)
def test_a_path_outside_every_home_is_refused(homes, tmp_path, outside):
    problem = _validate.base_model_error(outside)
    assert problem is not None, outside
    assert "base_model" in problem
    problems = _validate.validate_train_inputs(_spec(outside, tmp_path))
    assert any("base_model" in p for p in problems), problems


def test_dot_dot_cannot_climb_out_of_a_home(homes):
    climb = str(homes["out"] / ".." / ".." / "etc")
    assert _validate.base_model_error(climb) is not None


def test_a_symlink_out_of_a_home_is_refused(homes, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = homes["out"] / "link"
    os.symlink(target, link)
    assert _validate.base_model_error(str(link)) is not None


def test_the_refusal_is_the_same_sentence_whether_the_target_exists(homes, tmp_path):
    existing = tmp_path / "real"
    existing.mkdir()
    missing = tmp_path / "missing"
    said_of_existing = (_validate.base_model_error(str(existing)) or "").replace(str(existing), "<path>")
    said_of_missing = (_validate.base_model_error(str(missing)) or "").replace(str(missing), "<path>")
    assert said_of_existing == said_of_missing, "the sentence may echo the value, never what is on the disk"


def test_the_refusal_names_the_homes_and_the_hub_form_not_the_disk(homes):
    problem = _validate.base_model_error("/etc")
    assert problem is not None
    assert str(homes["out"]) in problem and str(homes["hub"]) in problem
    assert "org/name" in problem
    for word in ("exists", "not found", "no such"):
        assert word not in problem.lower()


@pytest.mark.parametrize("bad", ["a b/c", "org/name?x", "org//name", "/", "\x00", "org/name@", "@rev"])
def test_a_malformed_reference_is_refused(homes, bad):
    assert _validate.base_model_error(bad) is not None, repr(bad)


def test_an_empty_base_model_is_left_to_the_trainer(homes):
    """Whether the field is required is each trainer's call (Cosmos and GR00T require it)."""
    assert _validate.base_model_error("") is None


def test_the_dashboard_reads_the_same_rule(homes):
    """One owner: the dashboard's containment helpers are the training module's."""
    from strands_robots.dashboard import training as dash

    assert dash.looks_like_path is _validate.looks_like_path
    assert dash.hf_cache_root() == _validate.hf_cache_root()
