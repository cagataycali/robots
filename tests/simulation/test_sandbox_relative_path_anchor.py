"""Regression tests: a relative output path lands in the sandbox, not the cwd.

``validate_output_path`` resolved a relative path with ``Path.resolve()``, which
anchors to the process **cwd**. With a sandbox root configured, that made the same
call succeed or fail depending on where the process happened to be started:

    STRANDS_ROBOTS_RENDER_ROOT=/tmp/box
    cwd=/repo      render(output_path="shot.png")
      -> error: /repo/shot.png is outside the sandbox /tmp/box
    cwd=/tmp/box   render(output_path="shot.png")
      -> success

A sandbox root exists precisely to be where an unqualified path lands, so the obvious
call was rejected while an absolute path into the same directory worked. A relative
path is now anchored to the root.

The confinement itself is unchanged and re-verified here: ``..`` traversal, absolute
paths outside the root, tilde expansion outside it, and a symlink planted inside the
root but pointing out are all still refused. In guards-only mode (no root configured)
relative paths still resolve against the cwd, since there is no root to anchor to.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from strands_robots.simulation.safe_output import validate_output_path


def test_a_relative_path_anchors_to_the_sandbox_root(tmp_path) -> None:
    """The core defect: this used to resolve against the cwd."""
    root = tmp_path / "box"
    root.mkdir()
    resolved = validate_output_path("shot.png", sandbox_root=root, allow_abs=False)
    assert resolved == root / "shot.png"


def test_a_relative_subdirectory_also_anchors(tmp_path) -> None:
    root = tmp_path / "box"
    root.mkdir()
    resolved = validate_output_path("sub/shot.png", sandbox_root=root, allow_abs=False)
    assert resolved == root / "sub" / "shot.png"


def test_the_result_is_independent_of_the_cwd(tmp_path, monkeypatch) -> None:
    """Same call, two working directories, one answer."""
    root = tmp_path / "box"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(elsewhere)
    first = validate_output_path("shot.png", sandbox_root=root, allow_abs=False)
    monkeypatch.chdir(root)
    second = validate_output_path("shot.png", sandbox_root=root, allow_abs=False)
    assert first == second == root / "shot.png"


def test_an_absolute_path_inside_the_root_is_still_accepted(tmp_path) -> None:
    root = tmp_path / "box"
    root.mkdir()
    target = root / "abs.png"
    assert validate_output_path(str(target), sandbox_root=root, allow_abs=False) == target


def test_guards_only_mode_still_uses_the_cwd(tmp_path, monkeypatch) -> None:
    """With no root there is nothing to anchor to, so cwd remains correct."""
    monkeypatch.chdir(tmp_path)
    resolved = validate_output_path("shot.png", sandbox_root=None, allow_abs=True)
    assert resolved == tmp_path.resolve() / "shot.png"


@pytest.mark.parametrize("escape", ["../escape.png", "a/../../escape.png"])
def test_traversal_is_still_refused(tmp_path, escape) -> None:
    root = tmp_path / "box"
    root.mkdir()
    with pytest.raises(ValueError, match="traversal"):
        validate_output_path(escape, sandbox_root=root, allow_abs=False)


def test_an_absolute_path_outside_the_root_is_still_refused(tmp_path) -> None:
    root = tmp_path / "box"
    root.mkdir()
    outside = tmp_path / "outside.png"
    with pytest.raises(ValueError, match="outside the sandbox"):
        validate_output_path(str(outside), sandbox_root=root, allow_abs=False)


def test_tilde_outside_the_root_is_still_refused(tmp_path) -> None:
    root = tmp_path / "box"
    root.mkdir()
    with pytest.raises(ValueError, match="outside the sandbox"):
        validate_output_path("~/escape.png", sandbox_root=root, allow_abs=False)


def test_a_symlink_planted_inside_the_root_is_still_refused(tmp_path) -> None:
    """The anchoring must not turn a relative name into a symlink follow.

    Matched on ``refusing to follow`` rather than ``symlink``: pytest derives
    ``tmp_path`` from the test name, so every path in this test's messages
    contains the word "symlink" and any rejection at all would satisfy it - the
    guard has to be the one that fires, not the confinement check downstream.
    """
    root = tmp_path / "box"
    root.mkdir()
    target = tmp_path / "evil.png"
    target.write_text("original")
    os.symlink(target, root / "link.png")

    with pytest.raises(ValueError, match="refusing to follow"):
        validate_output_path("link.png", sandbox_root=root, allow_abs=False)
    assert target.read_text() == "original"


def test_a_symlink_pointing_back_inside_the_root_is_also_refused(tmp_path) -> None:
    """Confinement cannot catch this one - only the symlink guard can.

    A link inside the root whose target is ALSO inside the root passes the
    "resolved path is under the root" check, so it was followed silently while
    the same link pointing outside was refused (for the wrong reason).
    """
    root = tmp_path / "box"
    root.mkdir()
    target = root / "real.png"
    target.write_text("original")
    os.symlink(target, root / "link.png")

    with pytest.raises(ValueError, match="refusing to follow"):
        validate_output_path("link.png", sandbox_root=root, allow_abs=False)
    assert target.read_text() == "original"


def test_allow_abs_still_bypasses_confinement(tmp_path) -> None:
    """The documented opt-in escape hatch is unaffected."""
    root = tmp_path / "box"
    root.mkdir()
    outside = tmp_path / "outside.png"
    assert validate_output_path(str(outside), sandbox_root=root, allow_abs=True) == outside


def test_relative_anchoring_applies_under_allow_abs_too(tmp_path) -> None:
    """A relative path has no absolute intent, so it still belongs in the root."""
    root = tmp_path / "box"
    root.mkdir()
    assert validate_output_path("shot.png", sandbox_root=root, allow_abs=True) == root / "shot.png"


def test_a_resolved_root_is_used_for_comparison(tmp_path) -> None:
    """A symlinked root must still admit its own relative paths."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    resolved = validate_output_path("shot.png", sandbox_root=Path(link), allow_abs=False)
    assert resolved.name == "shot.png"
    assert resolved.parent.resolve() == real.resolve()
