"""A camera index this machine cannot have is refused BEFORE the arm is despawned.

reconfigure_cameras is a despawn+spawn. Every check that happens after the despawn is paid for with the
operator's running robot, so anything knowable up front must be judged up front. "index 7 on a machine with
3 capture devices" is knowable from the enumerated roster's COUNT alone - no device is opened, which the
supervisor law forbids for indices that are streaming.
"""

from __future__ import annotations

from strands_robots.dashboard.device_manager import DeviceManager, indices_beyond_roster


def test_an_index_past_the_roster_count_is_named_with_its_camera() -> None:
    assert indices_beyond_roster({"wrist": {"index_or_path": 7}}, 3) == {"wrist": 7}


def test_indices_inside_the_count_are_allowed_even_though_the_order_is_unknown() -> None:
    # The roster's ORDER is not evidence about which camera an index is (Continuity cameras renumber),
    # so an in-range index must never be refused here on the strength of a name.
    assert indices_beyond_roster({"top": {"index_or_path": 0}, "wrist": {"index_or_path": 2}}, 3) == {}


def test_a_negative_index_is_impossible_too() -> None:
    assert indices_beyond_roster({"top": {"index_or_path": -1}}, 3) == {"top": -1}


def test_an_empty_roster_refuses_nothing() -> None:
    # Enumeration failing (no ffmpeg, unsupported platform) is not evidence that a camera is absent.
    assert indices_beyond_roster({"wrist": {"index_or_path": 9}}, 0) == {}
    assert indices_beyond_roster({"wrist": {"index_or_path": 9}}, -1) == {}


def test_paths_and_bare_ints_are_handled_the_way_the_config_allows_them() -> None:
    # lerobot's shape allows a bare value as well as a mapping; a PATH cannot be compared to a count.
    assert indices_beyond_roster({"a": 5, "b": "/dev/video0", "c": True}, 3) == {"a": 5}


def test_reconfigure_refuses_an_impossible_index_without_touching_the_running_peer(monkeypatch) -> None:
    """The whole point: the arm keeps running, and no despawn is attempted."""
    dm = DeviceManager.__new__(DeviceManager)
    monkeypatch.setattr(DeviceManager, "_camera_names", lambda self, refresh=False: [{"index": 0}, {"index": 1}])

    def _never(*_a, **_k):  # pragma: no cover - the assertion is that this is unreachable
        raise AssertionError("despawn must not happen when the config cannot work")

    monkeypatch.setattr(DeviceManager, "despawn", _never)
    monkeypatch.setattr(DeviceManager, "spawn", _never)

    result = dm.reconfigure_cameras("so101-arm-1", {"wrist": {"index_or_path": 5}})

    assert "error" in result and not result.get("reconfigured")
    assert "index 5" in result["error"]
    assert "2 capture device" in result["error"]  # says what it counted
    assert "left running and untouched" in result["error"]  # says what it did NOT do
