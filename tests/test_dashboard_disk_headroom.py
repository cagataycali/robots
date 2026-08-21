"""Q92: the recording preflight can see the disk, and says so without crying wolf."""

from __future__ import annotations

from pathlib import Path

from strands_robots.dashboard import disk_headroom as dh


def test_a_comfortable_volume_says_nothing():
    assert dh.headroom_verdict(free_mb=200 * 1024, total_mb=900 * 1024) is None
    # Exactly at the threshold is still comfortable - the boundary belongs to the quiet side.
    assert dh.headroom_verdict(free_mb=dh.TIGHT_MB) is None


def test_no_reading_produces_no_verdict():
    # The law every check in this dashboard obeys: absent evidence cannot block or warn.
    assert dh.headroom_verdict(free_mb=None) is None
    assert dh.headroom_verdict(free_mb=-1) is None


def test_tight_warns_with_the_number_and_never_refuses():
    v = dh.headroom_verdict(free_mb=8 * 1024, total_mb=900 * 1024, where="the dataset home")
    assert v is not None and v["level"] == "tight"
    assert "8.0Gi" in v["headline"] and "the dataset home" in v["headline"]
    assert "0.9%" in v["headline"]  # the share, because 8Gi of 100Gi is a different story
    assert "swap" in v["advice"]


def test_critical_names_the_consequence_at_train_time():
    v = dh.headroom_verdict(free_mb=900)
    assert v is not None and v["level"] == "critical"
    assert "900Mi" in v["headline"]
    # The point of this wording: the operator will otherwise meet this failure as a parquet error.
    assert "meta/info.json" in v["advice"] and "TRAIN" in v["advice"]


def test_free_space_reads_a_real_volume_and_walks_up_to_it(tmp_path: Path):
    missing = tmp_path / "not" / "created" / "yet"
    got = dh.free_space(missing)
    assert got["path"] == str(missing)
    assert Path(got["measured_at"]).exists()
    assert got["free_mb"] > 0 and got["total_mb"] >= got["free_mb"]


def test_free_space_never_raises_and_never_answers_for_the_wrong_volume():
    # It must not raise. It must also not walk a nonsense RELATIVE path up to "." and report the
    # volume this process is running in as if it were the dataset's - a wrong number that looks
    # exactly like a right one. This assertion exists because the first version did that.
    assert dh.free_space("\0not-a-path") == {}
    assert dh.free_space("relative/datasets") == {}


# --- the wiring: the record session document carries it, idle AND recording ----------------------
class _FakeWorker:
    def session(self):
        return {"dataset": "local/x", "episodes": 3, "fps_notice": None}


def _manager():
    from strands_robots.dashboard.record_api import RecordController

    m = RecordController.__new__(RecordController)
    import threading
    from pathlib import Path

    m._lock = threading.Lock()
    m._worker = None
    m._crumb = Path("/nonexistent/crumb.json")
    m._parked = []
    m._disk_cache = None
    m._disk_seen = None
    return m


def test_the_idle_session_carries_a_tight_disk(monkeypatch):
    m = _manager()
    monkeypatch.setattr(dh, "free_space", lambda *a, **k: {"free_mb": 6 * 1024, "total_mb": 900 * 1024})
    got = m.session()
    assert got["disk_notice"]["level"] == "tight"
    assert "6.0Gi" in got["disk_notice"]["headline"]


def test_a_recording_session_carries_it_too_because_Q91_was_a_rate(monkeypatch):
    # The whole point: the volume lost ~2Gi/h to swap, so a session comfortable at open can be in
    # trouble by episode 20. A check that ran only at open would miss exactly that.
    m = _manager()
    m._worker = _FakeWorker()
    monkeypatch.setattr(dh, "free_space", lambda *a, **k: {"free_mb": 500, "total_mb": 900 * 1024})
    got = m.session()
    assert got["disk_notice"]["level"] == "critical"
    assert got["dataset"] == "local/x", "the worker's own document must survive intact"


def test_a_healthy_volume_leaves_the_key_ABSENT(monkeypatch):
    # Absent, not null: every other *_notice on this document works that way, and a null would make
    # the UI render an empty warning row.
    m = _manager()
    monkeypatch.setattr(dh, "free_space", lambda *a, **k: {"free_mb": 400 * 1024, "total_mb": 900 * 1024})
    assert "disk_notice" not in m.session()


def test_an_unreadable_volume_cannot_break_the_session(monkeypatch):
    m = _manager()
    monkeypatch.setattr(dh, "free_space", lambda *a, **k: {})
    assert "disk_notice" not in m.session()


def test_the_reading_is_cached_so_a_1hz_poll_is_not_a_1hz_statfs(monkeypatch):
    m = _manager()
    calls = []

    def counted(*a, **k):
        calls.append(1)
        return {"free_mb": 6 * 1024, "total_mb": 900 * 1024}

    monkeypatch.setattr(dh, "free_space", counted)
    for _ in range(5):
        m.session()
    assert len(calls) == 1
