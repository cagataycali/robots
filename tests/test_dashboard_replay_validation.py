"""Q5: a replay that cannot start must be refused before it has a pid.

``/api/replay`` answered ``200 {"pid": ...}`` for a negative episode and a
nonexistent dataset root - the truth arrived seconds later as a dead child in a
log nobody was reading. Everything knowable without a network call is now judged
by ``validate_replay()`` BEFORE Popen. Hub existence is deliberately not probed
(a network round-trip in a request path; an offline dashboard must still replay
from cache), so a bogus-but-well-formed hub id still spawns - the fleet card and
child log stay the honest surface for that case, and these tests pin the line.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.device_manager import DeviceManager, validate_replay


class TestValidateReplay:
    def test_a_well_formed_hub_request_passes(self) -> None:
        assert validate_replay("lerobot/pusht", 0) is None

    def test_a_bare_local_dataset_name_passes(self) -> None:
        assert validate_replay("collected-blocks", 3) is None

    @pytest.mark.parametrize("episode", [-1, -5])
    def test_a_negative_episode_is_refused_as_a_list_position(self, episode: int) -> None:
        bad = validate_replay("lerobot/pusht", episode)
        assert bad is not None and ">= 0" in bad["error"]

    @pytest.mark.parametrize("episode", ["3", 2.5, None, True, [1]])
    def test_a_non_integer_episode_is_refused_by_type_not_coerced(self, episode: object) -> None:
        # int("3") would work and int(2.5) would silently floor - both hide
        # what the client actually sent. bool is an int in Python; an episode
        # of True is a client bug, not episode 1.
        bad = validate_replay("lerobot/pusht", episode)
        assert bad is not None and "integer" in bad["error"]

    @pytest.mark.parametrize("speed", [0, -1.0, float("inf"), float("nan")])
    def test_speed_must_be_finite_and_positive(self, speed: float) -> None:
        bad = validate_replay("lerobot/pusht", 0, speed=speed)
        assert bad is not None and "speed" in bad["error"]

    def test_speed_accepts_an_honest_number(self) -> None:
        assert validate_replay("lerobot/pusht", 0, speed=0.5) is None

    @pytest.mark.parametrize("rid", ["", "  ", None, 42, "a b/c", "/leading", "org//name", "org/name/extra"])
    def test_a_malformed_repo_id_is_refused(self, rid: object) -> None:
        assert validate_replay(rid, 0) is not None

    def test_a_root_that_does_not_exist_is_refused(self, tmp_path) -> None:
        bad = validate_replay("local/set", 0, root=str(tmp_path / "nope"))
        assert bad is not None and "does not exist" in bad["error"]

    def test_a_root_that_exists_passes(self, tmp_path) -> None:
        assert validate_replay("local/set", 0, root=str(tmp_path)) is None

    def test_no_root_means_hub_or_cache_and_is_not_our_call(self) -> None:
        # The deliberate line: existence on the hub is not probed here.
        assert validate_replay("nobody/nothing-zz", 0) is None


class TestReplayRefusesBeforePopen:
    def test_a_negative_episode_never_reaches_popen(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))

        def boom(*a, **k):  # noqa: ANN002, ANN003
            raise AssertionError("Popen was reached for an invalid replay")

        import strands_robots.dashboard.device_manager as mod

        monkeypatch.setattr(mod.subprocess, "Popen", boom)
        result = dm.replay("lerobot/pusht", episode=-5)
        assert "error" in result and "pid" not in result
        assert dm.robots == {}, "a refused replay must not register a managed entry"

    def test_a_missing_root_never_reaches_popen(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        import strands_robots.dashboard.device_manager as mod

        monkeypatch.setattr(
            mod.subprocess, "Popen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("Popen reached")),
        )
        result = dm.replay("local/set", episode=0, root=str(tmp_path / "missing"))
        assert "error" in result and "does not exist" in result["error"]
