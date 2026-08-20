"""A taken dataset name is refused BEFORE the arms are touched (Q39).

Every failure after the parking step in RecordController.open reports through
``could not open the arms: {exc}`` — so a dataset whose directory already exists sent the operator
to check cables and USB ports for what is a one-word rename, after both arms had been despawned and
respawned for nothing.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.dashboard.dataset_check import record_target_verdict


def _home(monkeypatch, path) -> None:
    """Point dataset resolution at a temp home.

    NOT via the environment: ``_lerobot_home()`` prefers lerobot's own ``HF_LEROBOT_HOME``
    CONSTANT, which lerobot resolved when it was first imported - so setenv here changes nothing
    and the test would pass on a false negative (no dataset found => no refusal). Patching the
    function is the honest lever, and it is also the reason production must set that variable
    before the dashboard starts, not after.
    """
    from pathlib import Path

    from strands_robots import dataset_recorder

    monkeypatch.setattr(dataset_recorder, "_lerobot_home", lambda: Path(str(path)))


class TestTheVerdict:
    def test_no_name_at_all(self) -> None:
        for empty in ("", "   ", None):
            v = record_target_verdict(empty)  # type: ignore[arg-type]
            assert v and "dataset name is required" in v
            assert "repo_id" in v, "the refusal has to say what the name becomes"

    def test_a_free_name_is_not_refused(self) -> None:
        assert record_target_verdict("local/new-one", exists=False) is None

    def test_an_existing_dataset_with_episodes_names_what_is_at_stake(self) -> None:
        v = record_target_verdict("local/good", exists=True, has_meta=True, episodes=42)
        assert v is not None
        assert "42 recorded episode(s)" in v
        # BOTH halves of the truth: recording refuses, and overwriting destroys.
        assert "refuse" in v and "destroy" in v
        assert "Pick another name" in v

    def test_an_interrupted_session_s_leftovers_read_differently(self) -> None:
        # Same FileExistsError, different next action: there is nothing to lose here, so the
        # sentence must not imply the operator is about to destroy recorded work.
        v = record_target_verdict("local/half", exists=True, has_meta=True, episodes=0)
        assert v is not None
        assert "no recorded episodes" in v and "interrupted session" in v
        assert "destroy" not in v

    def test_a_non_dataset_directory_is_its_own_case(self) -> None:
        v = record_target_verdict("local/notes", exists=True, has_meta=False, non_empty=True)
        assert v is not None
        assert "not a dataset" in v
        assert "nothing here will delete files for you" in v

    def test_an_empty_directory_in_the_way_is_not_a_refusal(self) -> None:
        # An empty directory is what a resolve-then-mkdir dance leaves behind and LeRobot is happy
        # to write into it; refusing would invent a problem.
        assert record_target_verdict("local/x", exists=True, has_meta=False, non_empty=False) is None

    def test_an_unknown_episode_count_does_not_claim_a_number(self) -> None:
        v = record_target_verdict("local/x", exists=True, has_meta=True, episodes=None)
        assert v is not None and "no recorded episodes" in v


class TestTheFactsAreReadDefensively:
    def test_an_unreadable_home_yields_no_facts_rather_than_an_error(self, monkeypatch) -> None:
        from strands_robots.dashboard import record_api

        _home(monkeypatch, "/dev/null/nope")
        assert record_api._target_facts("local/whatever") in ({}, {"exists": False})

    def test_it_reports_a_real_dataset_on_disk(self, tmp_path, monkeypatch) -> None:
        from strands_robots.dashboard import record_api

        _home(monkeypatch, tmp_path)
        d = tmp_path / "local" / "taken" / "meta"
        d.mkdir(parents=True)
        (d / "info.json").write_text(json.dumps({"total_episodes": 7, "fps": 30}))
        facts = record_api._target_facts("local/taken")
        assert facts["exists"] is True and facts["has_meta"] is True and facts["episodes"] == 7
        # ...and the two halves compose into the sentence the operator sees.
        v = record_target_verdict("local/taken", **facts)
        assert v is not None and "7 recorded episode(s)" in v

    def test_an_empty_name_never_touches_the_disk(self) -> None:
        from strands_robots.dashboard import record_api

        assert record_api._target_facts("  ") == {}


def test_open_refuses_before_parking_any_arm(monkeypatch, tmp_path) -> None:
    """The point of the whole exercise: no despawn, no respawn, no 'could not open the arms'."""
    from fastapi import HTTPException

    from strands_robots.dashboard import record_api

    _home(monkeypatch, tmp_path)
    meta = tmp_path / "local" / "taken" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 3, "fps": 30}))

    class Devices:
        def __init__(self) -> None:
            self.despawned: list[str] = []

        def despawn(self, peer_id: str) -> None:  # pragma: no cover - must not be reached
            self.despawned.append(peer_id)

    devices = Devices()
    ctl = record_api.RecordController(devices=devices, backend_factory=lambda **_: object())
    with pytest.raises(HTTPException) as err:
        ctl.open({"dataset": "local/taken", "task": "t", "leader": "a", "follower": "b"})

    assert err.value.status_code == 409
    assert "3 recorded episode(s)" in str(err.value.detail)
    assert "could not open the arms" not in str(err.value.detail)
    assert devices.despawned == [], "the arms were parked for a name collision"
