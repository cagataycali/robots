"""Robot(...) 404 surfaces a swallowed user-overlay parse error.

The overlay getter catches ``json.JSONDecodeError`` so ``get_robot()`` can
read a corrupt overlay without raising (regression: see
``test_user_registry.py::TestPersistence::test_corrupted_json_returns_empty``).
That silence hides the user's actual problem - a one-character typo in a file
they can fix - behind the generic "Unknown robot" 404.

The ``Robot(...)`` 404 reads :func:`strands_robots.registry._overlay.last_parse_error`
and names the file + parse error so the user is pointed at the right place.
"""
from __future__ import annotations

import pathlib

import pytest

from strands_robots.registry._overlay import last_parse_error, user_registry_path
from strands_robots.registry.loader import invalidate_cache


@pytest.fixture
def _reset_overlay_state(monkeypatch, tmp_path):
    """Fresh ``STRANDS_BASE_DIR`` and clean overlay state before/after each test."""
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    invalidate_cache()
    # Clear any stale last_parse_error from a previous test file.
    from strands_robots.registry import _overlay as _ov

    _ov._last_parse_error[0] = None
    yield
    _ov._last_parse_error[0] = None
    invalidate_cache()


def _trigger_overlay_parse(path: pathlib.Path) -> None:
    """Read the overlay once to run the parser and (if broken) record the error."""
    from strands_robots.registry._overlay import parse_user_robots, user_registry_source

    parse_user_robots(user_registry_source())


class TestOverlayParseErrorIsExposed:
    def test_last_parse_error_is_none_without_overlay(self, _reset_overlay_state):
        _trigger_overlay_parse(user_registry_path())
        assert last_parse_error() is None

    def test_last_parse_error_is_none_on_valid_overlay(self, _reset_overlay_state, tmp_path):
        (tmp_path / "user_robots.json").write_text('{"robots": {}}')
        _trigger_overlay_parse(user_registry_path())
        assert last_parse_error() is None

    def test_last_parse_error_names_file_and_reason_on_syntax_error(
        self, _reset_overlay_state, tmp_path
    ):
        p = tmp_path / "user_robots.json"
        p.write_text('{"robots": {"my_bot": {"category": "arm",,}}}')  # trailing comma
        _trigger_overlay_parse(user_registry_path())
        err = last_parse_error()
        assert err is not None
        file_path, reason = err
        assert file_path == str(p)
        # The parser's JSONDecodeError reason is passed through verbatim.
        assert "line" in reason and "column" in reason

    def test_last_parse_error_clears_after_a_subsequent_valid_parse(
        self, _reset_overlay_state, tmp_path
    ):
        p = tmp_path / "user_robots.json"
        p.write_text('{"robots": {"my_bot": {},,}}')  # broken
        _trigger_overlay_parse(user_registry_path())
        assert last_parse_error() is not None
        p.write_text('{"robots": {}}')  # fixed
        invalidate_cache()
        _trigger_overlay_parse(user_registry_path())
        assert last_parse_error() is None


class TestRobotRefusalSurfacesOverlayParseError:
    def test_unknown_robot_404_names_the_broken_overlay(
        self, _reset_overlay_state, tmp_path
    ):
        (tmp_path / "user_robots.json").write_text(
            '{"robots": {"my_bot": {"category": "arm",,}}}'  # trailing comma
        )
        _trigger_overlay_parse(user_registry_path())  # populate last_parse_error
        from strands_robots import Robot

        with pytest.raises(ValueError, match=r"Unknown robot") as excinfo:
            Robot("my_bot", mesh=False)
        msg = str(excinfo.value)
        # The user is pointed at the exact file to fix.
        assert str(user_registry_path()) in msg
        assert "failed to parse" in msg
        # The rest of the 404 (did-you-mean, list_robots hint) still fires.
        assert "list_robots" in msg

    def test_unknown_robot_404_unchanged_when_overlay_is_absent(
        self, _reset_overlay_state
    ):
        # No overlay file at all -> no parse_error, no mention of user_robots.json.
        _trigger_overlay_parse(user_registry_path())
        from strands_robots import Robot

        with pytest.raises(ValueError, match=r"Unknown robot") as excinfo:
            Robot("absolutely_not_a_robot_xyz", mesh=False)
        msg = str(excinfo.value)
        assert "user_robots.json" not in msg
        assert "failed to parse" not in msg

    def test_unknown_robot_404_unchanged_when_overlay_parses_cleanly(
        self, _reset_overlay_state, tmp_path
    ):
        (tmp_path / "user_robots.json").write_text('{"robots": {}}')
        _trigger_overlay_parse(user_registry_path())
        from strands_robots import Robot

        with pytest.raises(ValueError, match=r"Unknown robot") as excinfo:
            Robot("absolutely_not_a_robot_xyz", mesh=False)
        msg = str(excinfo.value)
        assert "failed to parse" not in msg
