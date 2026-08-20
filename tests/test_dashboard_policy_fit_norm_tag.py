"""An undeclared norm_tag is refused while the form is open, not after the arm is torqued."""
from __future__ import annotations

import json

from strands_robots.dashboard.checkpoints import _declared_norm_tags
from strands_robots.dashboard.policy_fit import policy_fit

FEATS = {"observation.state": {"type": "STATE", "shape": [6]}}
OUT = {"action": {"type": "ACTION", "shape": [6]}}
ARM = {"joints": [f"j{i}" for i in range(6)], "cameras": []}


def _fit(**kw):
    return policy_fit(input_features=FEATS, output_features=OUT, joints=ARM["joints"], **kw)


def test_an_undeclared_tag_blocks_and_names_what_is_declared() -> None:
    v = _fit(norm_tag="mean_std", declared_norm_tags=["min_max", "q99"])
    assert v["ok"] is False and v["blocking"] is True
    p = next(x for x in v["problems"] if x["field"] == "norm_tag")
    assert "min_max, q99" in p["text"], "the operator needs the choices, not just a refusal"
    assert "wrong statistics" in p["text"] and "real arm" in p["text"], "name the physical consequence"
    assert "min_max, q99" in p["remedy"]


def test_a_declared_tag_is_recorded_as_CHECKED_not_silent() -> None:
    v = _fit(norm_tag="min_max", declared_norm_tags=["min_max", "q99"])
    assert [x for x in v["problems"] if x["field"] == "norm_tag"] == []
    assert "norm_tag" in v["checked"], "a quiet answer must read as verified, not as unexamined"


def test_no_declared_tags_is_no_evidence_never_a_refusal() -> None:
    # An older checkpoint ships no norm_stats.json; treating that silence as a mismatch would block
    # runs that have always worked.
    v = _fit(norm_tag="mean_std", declared_norm_tags=[])
    assert v["ok"] is True and [x for x in v["problems"] if x["field"] == "norm_tag"] == []
    assert "norm_tag" not in v["checked"]


def test_no_tag_requested_is_not_checked() -> None:
    v = _fit(norm_tag=None, declared_norm_tags=["min_max"])
    assert "norm_tag" not in v["checked"] and v["ok"] is True


def test_declared_tags_are_read_from_the_stats_file(tmp_path) -> None:
    (tmp_path / "norm_stats.json").write_text(json.dumps({"q99": {}, "min_max": {}}))
    assert _declared_norm_tags(tmp_path) == ["min_max", "q99"]


def test_an_unreadable_or_absent_stats_file_yields_no_evidence(tmp_path) -> None:
    assert _declared_norm_tags(tmp_path) == []
    (tmp_path / "norm_stats.json").write_text("{not json")
    assert _declared_norm_tags(tmp_path) == []
    (tmp_path / "norm_stats.json").write_text("[1, 2]")
    assert _declared_norm_tags(tmp_path) == [], "a list declares no tags"
