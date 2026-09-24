"""Behavior tests for the ``download_assets`` agent tool.

Covers the four documented actions (``list``, ``status``, ``download``,
unknown) plus the error path. The tool is a thin wrapper around
:mod:`strands_robots.assets.download`; the underlying download logic is
mocked so these tests run hardware- and network-free, asserting on the
tool's parsing/formatting contract rather than implementation details.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from strands_robots.tools.download_assets import download_assets

_MOD = "strands_robots.tools.download_assets"


def test_list_action_returns_robot_table() -> None:
    """``action='list'`` returns the formatted registry table."""
    with patch(f"{_MOD}.format_robot_table", return_value="ROBOT-TABLE"):
        result = download_assets(action="list")
    assert result["status"] == "success"
    assert "ROBOT-TABLE" in result["content"][0]["text"]


def test_status_action_summarizes_availability() -> None:
    """``action='status'`` counts available vs missing and shows the cache dir."""
    robots_info = [
        {"name": "so100", "category": "arm", "description": "arm", "available": True},
        {"name": "panda", "category": "arm", "description": "arm", "available": False},
    ]
    with (
        patch(f"{_MOD}.list_available_robots", return_value=robots_info),
        patch(f"{_MOD}.get_user_assets_dir", return_value="/tmp/assets"),
    ):
        result = download_assets(action="status")
    text = result["content"][0]["text"]
    assert result["status"] == "success"
    assert "1 available, 1 missing" in text
    assert "/tmp/assets" in text
    assert "so100" in text and "panda" in text


def test_status_action_marks_each_row_available_or_missing() -> None:
    """Each robot row carries a per-row availability marker.

    Regression: the marker was an empty-both ``'' if r['available'] else ''``
    ternary (an emoji-stripping fossil), so downloaded and missing robots
    rendered identically and the per-row status conveyed nothing. Assert the
    marker is present and correctly maps availability to the robot on its row.
    """
    robots_info = [
        {"name": "so100", "category": "arm", "description": "arm", "available": True},
        {"name": "panda", "category": "arm", "description": "arm", "available": False},
    ]
    with (
        patch(f"{_MOD}.list_available_robots", return_value=robots_info),
        patch(f"{_MOD}.get_user_assets_dir", return_value="/tmp/assets"),
    ):
        result = download_assets(action="status")
    rows = {
        name: next(ln for ln in result["content"][0]["text"].splitlines() if name in ln) for name in ("so100", "panda")
    }
    # Available and missing rows are visually distinguishable.
    assert rows["so100"] != rows["panda"].replace("panda", "so100")
    # ...and the marker maps to the correct robot.
    assert "[ok]" in rows["so100"] and "[--]" not in rows["so100"]
    assert "[--]" in rows["panda"] and "[ok]" not in rows["panda"]


def test_download_action_parses_names_and_reports_counts() -> None:
    """``action='download'`` splits comma names and surfaces result counts."""
    fake_result = {
        "downloaded": 2,
        "skipped": 1,
        "failed": 0,
        "method": "robot_descriptions",
        "assets_dir": "/tmp/assets",
    }
    with patch(f"{_MOD}.download_robots", return_value=fake_result) as mock_dl:
        result = download_assets(action="download", robots="so100, panda ", category="arm", force=True)
    mock_dl.assert_called_once_with(names=["so100", "panda"], category="arm", force=True)
    text = result["content"][0]["text"]
    assert result["status"] == "success"
    assert "Downloaded: 2, Skipped: 1, Failed: 0" in text
    assert "robot_descriptions" in text


def test_download_action_with_no_names_passes_none() -> None:
    """Omitting ``robots`` downloads all (names=None)."""
    fake_result = {"downloaded": 1, "skipped": 0, "failed": 0, "method": "git", "assets_dir": "/d"}
    with patch(f"{_MOD}.download_robots", return_value=fake_result) as mock_dl:
        download_assets(action="download")
    mock_dl.assert_called_once_with(names=None, category=None, force=False)


def test_download_action_lists_failed_details() -> None:
    """Failed downloads are itemized in the output."""
    fake_result = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 1,
        "method": "git",
        "assets_dir": "/d",
        "failed_details": {"badbot": "clone failed"},
    }
    with patch(f"{_MOD}.download_robots", return_value=fake_result):
        result = download_assets(action="download", robots="badbot")
    text = result["content"][0]["text"]
    assert "badbot" in text and "clone failed" in text


def test_unknown_action_returns_error() -> None:
    """An unrecognized action is rejected with the valid-action list."""
    result = download_assets(action="bogus")
    assert result["status"] == "error"
    assert "Unknown action" in result["content"][0]["text"]


def test_underlying_exception_is_caught_and_reported() -> None:
    """Exceptions from the download layer become a structured error result."""
    with patch(f"{_MOD}.download_robots", side_effect=RuntimeError("boom")):
        result = download_assets(action="download", robots="so100")
    assert result["status"] == "error"
    assert "boom" in result["content"][0]["text"]


def _library_result(**overrides: object) -> dict[str, object]:
    """A ``download_robots`` result that fetched nothing, overridden per case.

    The three counts and ``unknown_names`` are on every return path, so a case
    only states what distinguishes it.
    """
    return {"downloaded": 0, "skipped": 0, "failed": 0, "unknown_names": [], **overrides}


_NO_MATCH = "No matching robots found."
_ALL_PRESENT = "All {n} robots already have assets. Use force=True to re-download."


@pytest.mark.parametrize(
    ("case", "library_result", "expected_status", "must_name"),
    [
        # An unlisted name matches nothing, so the library early-returns before it
        # reaches a download: no ``method``, no ``assets_dir``, and the zeros a
        # caller cannot tell from a completed no-op.
        (
            "an unknown name",
            _library_result(unknown_names=["no_such_robot"], message=_NO_MATCH),
            "error",
            "no_such_robot",
        ),
        # The same zeros with nothing to name: an unmatched category is the empty
        # selection ``unknown_names`` cannot describe, so the count grades it.
        ("an unmatched category", _library_result(message=_NO_MATCH), "error", _NO_MATCH),
        (
            "a failed clone",
            _library_result(failed=1, failed_details={"badbot": "clone failed"}, method="git clone", assets_dir="/d"),
            "error",
            "clone failed",
        ),
        # Why the named list is read beside the count: one real robot carries
        # ``skipped == 1``, so this selection did fetch something and is still not
        # the selection the caller asked for.
        (
            "an unknown name beside a present one",
            _library_result(
                skipped=1,
                skipped_names=["so100"],
                unknown_names=["no_such_robot"],
                message=_ALL_PRESENT.format(n=1),
            ),
            "error",
            "no_such_robot",
        ),
        # Did not over-fire: a selection already on disk fetched nothing on purpose
        # and reports why, not two placeholders.
        (
            "a selection already present",
            _library_result(skipped=2, skipped_names=["so100", "panda"], message=_ALL_PRESENT.format(n=2)),
            "success",
            "already have assets",
        ),
        (
            "a completed download",
            _library_result(downloaded=2, skipped=1, method="robot_descriptions", assets_dir="/tmp/assets"),
            "success",
            "robot_descriptions",
        ),
    ],
)
def test_a_download_that_fetched_nothing_is_not_a_success(
    case: str,
    library_result: dict[str, object],
    expected_status: str,
    must_name: str,
) -> None:
    """Every way of fetching nothing is graded, and the verdict says which one.

    ``download`` reported ``status="success"`` with ``Downloaded: 0, Skipped: 0,
    Failed: 0`` for an unknown name, for a failed clone and for a selection that
    matched no robot - the last reachable from an unmatched ``category=``. The
    text under those verdicts rendered ``Method: ?`` and ``Assets: ?``, because a
    result that never reached a download carries neither key.
    """
    with patch(f"{_MOD}.download_robots", return_value=library_result):
        result = download_assets(action="download", robots="whatever")

    text = result["content"][0]["text"]
    assert result["status"] == expected_status, f"{case}: {text}"
    assert must_name in text, f"{case}: {text}"
    # No placeholder stands in for the reason on any path.
    assert "?" not in text, f"{case}: {text}"


@pytest.mark.parametrize(
    ("robots", "category", "must_name"),
    [
        ("no_such_robot", None, "no_such_robot"),
        (None, "no_such_category", "No matching robots found."),
    ],
    ids=["an unlisted name", "an unmatched category"],
)
def test_an_empty_selection_is_graded_on_the_librarys_own_result(
    tmp_path: Path, robots: str | None, category: str | None, must_name: str
) -> None:
    """The verdict is graded on the result ``download_robots`` really returns.

    The table above states the library's result shapes rather than observing
    them, so it would keep passing if ``download_robots`` stopped reporting
    ``unknown_names`` or its ``message``. These two selectors match no robot, so
    the real function early-returns before any clone - network-free by
    construction - and the keys the verdict reads are the ones it produced.
    """
    download = importlib.import_module("strands_robots.assets.download")
    with patch.object(download, "get_user_assets_dir", lambda: tmp_path):
        result = download_assets(action="download", robots=robots, category=category)

    text = result["content"][0]["text"]
    assert result["status"] == "error", text
    assert "Downloaded: 0, Skipped: 0, Failed: 0" in text, text
    assert must_name in text, text
    assert "?" not in text, text
