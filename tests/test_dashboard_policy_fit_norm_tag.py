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
    p = next(x for x in v["problems"] if x["kind"] == "norm_tag")
    assert "min_max, q99" in p["detail"], "the operator needs the choices, not just a refusal"
    assert "wrong statistics" in p["detail"] and "real arm" in p["detail"], "name the consequence"


def test_a_declared_tag_is_recorded_as_CHECKED_not_silent() -> None:
    v = _fit(norm_tag="min_max", declared_norm_tags=["min_max", "q99"])
    assert [x for x in v["problems"] if x["kind"] == "norm_tag"] == []
    assert "norm_tag" in v["checked"], "a quiet answer must read as verified, not as unexamined"


def test_no_declared_tags_is_no_evidence_never_a_refusal() -> None:
    # An older checkpoint ships no norm_stats.json; treating that silence as a mismatch would block
    # runs that have always worked.
    v = _fit(norm_tag="mean_std", declared_norm_tags=[])
    assert v["ok"] is True and [x for x in v["problems"] if x["kind"] == "norm_tag"] == []
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


def test_every_problem_this_module_can_emit_has_the_keys_THE_SCREEN_RENDERS() -> None:
    """RunForm renders `p.detail` keyed by `p.kind`. A problem shaped any other way blocks the run
    with a BLANK line - refused, unexplained, which is worse than the mismatch it reports. This
    caught exactly that: the norm_tag problem first shipped as {field, text, remedy}."""
    verdicts = [
        _fit(norm_tag="mean_std", declared_norm_tags=["min_max"]),
        # a state dim that disagrees, an action dim that disagrees, a camera the peer lacks
        policy_fit(input_features={"observation.state": {"type": "STATE", "shape": [5]}},
                   output_features={"action": {"type": "ACTION", "shape": [2]}},
                   joints=ARM["joints"], cameras=["top"]),
        policy_fit(input_features={"observation.images.front": {"type": "VISUAL", "shape": [3, 8, 8]}},
                   output_features=OUT, joints=ARM["joints"], cameras=["top", "wrist"]),
    ]
    seen = [p for v in verdicts for p in v["problems"]]
    assert len(seen) >= 3, "the premise: these inputs really do produce problems to inspect"
    for p in seen:
        assert set(p) == {"kind", "detail"}, f"unrenderable problem shape: {sorted(p)}"
        assert p["kind"].strip() and p["detail"].strip(), "a blank refusal explains nothing"
