"""The record-session crumb is written and unlinked only inside ``~/.strands_dashboard``.

``STRANDS_DASH_RECORD_CRUMB`` was page-writable through ``POST /api/config`` and
``crumb_path`` honoured it verbatim, so a record session wrote a JSON file at any path
(creating its parents) and unlinked it when the session closed. The key is no longer
page-writable, and an override that resolves outside the dashboard's own directory, a
symlink out of it included, is ignored in favour of the default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.dashboard import config_api, record_crash


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    state = tmp_path / "home" / ".strands_dashboard"
    state.mkdir(parents=True)
    (state / "link-out").symlink_to(tmp_path / "elsewhere")
    return state.resolve()


@pytest.mark.parametrize(
    ("override", "inside"),
    [
        (None, None),
        ("~/.strands_dashboard/other.json", "other.json"),
        ("{tmp}/elsewhere/victim.json", None),
        ("~/.strands_dashboard/../victim.json", None),
        ("~/.strands_dashboard/link-out/victim.json", None),
        ("~/.strands_dashboard", None),
    ],
)
def test_the_crumb_never_leaves_the_dashboard_home(home, monkeypatch, override, inside):
    if override is None:
        monkeypatch.delenv("STRANDS_DASH_RECORD_CRUMB", raising=False)
    else:
        monkeypatch.setenv("STRANDS_DASH_RECORD_CRUMB", override.format(tmp=home.parent.parent))
    path = record_crash.crumb_path()
    assert path == home / (inside or "record_session.json")
    record_crash.write_crumb({"dataset": "d"})
    assert record_crash.read_crumb() is not None
    record_crash.clear_crumb()
    assert not path.exists()
    assert not (home.parent.parent / "elsewhere").exists(), "nothing written outside the home"


def test_the_page_cannot_write_the_crumb_path():
    assert config_api.env_entry_error("STRANDS_DASH_RECORD_CRUMB", "/tmp/x.json") is not None
