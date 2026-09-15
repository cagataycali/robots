"""``Simulation(render_dir=...)`` / ``Robot(name, render_dir=...)`` moves the render sandbox.

The model supplies ``output_path`` and cannot move the sandbox; the developer
who constructs the Simulation can, per instance, without an environment
variable set before the process started. The confinement rules are unchanged -
only the root they compare against comes from the constructor.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl as _requires_mujoco  # noqa: E402


@pytest.fixture
def process_sandbox(tmp_path, monkeypatch):
    """A process-wide sandbox the constructor must be able to override."""
    root = tmp_path / "process-renders"
    root.mkdir()
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(root))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)
    return root


def _saved_path(result: dict) -> str:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]["saved_path"]
    raise AssertionError(f"no json block in render result: {result}")


def test_render_dir_error_refuses_only_what_cannot_name_a_directory(tmp_path):
    from strands_robots.simulation.mujoco.rendering import render_dir_error

    assert render_dir_error(str(tmp_path)) is None
    assert render_dir_error(tmp_path) is None
    assert render_dir_error(str(tmp_path / "not-yet-created")) is None
    assert "got bool True" in render_dir_error(True)
    assert "got int 0" in render_dir_error(0)
    assert "got list" in render_dir_error(["x"])
    assert "empty path" in render_dir_error("")
    assert "empty path" in render_dir_error("   ")


def test_render_dir_error_describes_a_value_whose_repr_raises():
    """The message about an unusable argument must not be the thing that fails.

    ``refusal_repr`` is how every guard in the package renders a refused value,
    for this reason: a third-party type owes a guard nothing beyond the type
    test it failed, and rendering it must not raise on the one path that exists
    to answer a bad value with text.
    """
    from strands_robots.simulation.mujoco.rendering import render_dir_error

    class Unprintable:
        def __repr__(self):
            raise RuntimeError("repr is not available")

    message = render_dir_error(Unprintable())
    assert message is not None
    assert "render_dir must be a directory path" in message
    assert "Unprintable" in message


def test_resolve_render_dir_expands_and_normalizes(tmp_path, monkeypatch):
    from strands_robots.simulation.mujoco.rendering import resolve_render_dir

    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_render_dir("~/shots") == (tmp_path / "shots").resolve()
    assert resolve_render_dir(tmp_path / "a" / ".." / "b") == (tmp_path / "b").resolve()


def test_constructor_refuses_a_render_dir_that_is_not_a_path():
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    with pytest.raises(ValueError, match="render_dir must be a directory path"):
        MuJoCoSimEngine(render_dir=True)
    with pytest.raises(ValueError, match="render_dir must name a directory"):
        MuJoCoSimEngine(render_dir="")


def test_constructor_default_keeps_the_process_sandbox():
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    sim = MuJoCoSimEngine()
    try:
        assert sim.render_dir is None
    finally:
        sim.cleanup()


@_requires_mujoco
def test_bare_filename_lands_in_the_constructor_render_dir(tmp_path, process_sandbox):
    """``render_dir`` wins over ``STRANDS_ROBOTS_RENDER_ROOT``; the dir is created on first use."""
    from strands_robots import Robot

    mine = tmp_path / "mine" / "shots"
    sim = Robot("so101", mesh=False, render_dir=str(mine))
    try:
        assert sim.render_dir == mine.resolve()
        result = sim.render(camera_name="default", width=64, height=48, output_path="frame.png")
        assert result["status"] == "success", result
        assert _saved_path(result) == str(mine.resolve() / "frame.png")
        assert (mine / "frame.png").is_file()
        assert not any(process_sandbox.iterdir()), "the process sandbox must stay untouched"
    finally:
        sim.cleanup()


@_requires_mujoco
def test_absolute_path_under_the_constructor_render_dir_is_accepted(tmp_path, process_sandbox):
    """The owner asked for a file in their directory; with render_dir on that directory the model can deliver it."""
    from strands_robots import Robot

    mine = tmp_path / "mine"
    sim = Robot("so101", mesh=False, render_dir=mine)
    try:
        target = mine / "views" / "front.png"
        result = sim.render(camera_name="default", width=64, height=48, output_path=str(target))
        assert result["status"] == "success", result
        assert Path(_saved_path(result)) == target.resolve()
        assert target.is_file()
    finally:
        sim.cleanup()


@_requires_mujoco
def test_absolute_path_outside_the_constructor_render_dir_is_refused_naming_it(tmp_path, process_sandbox):
    """Confinement is unchanged; the refusal quotes THIS Simulation's sandbox."""
    from strands_robots import Robot

    mine = tmp_path / "mine"
    sim = Robot("so101", mesh=False, render_dir=mine)
    try:
        elsewhere = tmp_path / "elsewhere.png"
        result = sim.render(camera_name="default", width=64, height=48, output_path=str(elsewhere))
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "outside the sandbox" in text
        assert str(mine.resolve()) in text
        assert "STRANDS_ROBOTS_RENDER_ALLOW_ABS" in text
        assert not elsewhere.exists()
        assert not any(process_sandbox.iterdir())
    finally:
        sim.cleanup()


@_requires_mujoco
def test_two_simulations_keep_separate_render_dirs(tmp_path, process_sandbox):
    from strands_robots.simulation import Simulation

    a = Simulation(tool_name="a_sim", render_dir=tmp_path / "a")
    b = Simulation(tool_name="b_sim", render_dir=tmp_path / "b")
    try:
        a.create_world()
        b.create_world()
        ra = a.render(camera_name="default", width=32, height=24, output_path="f.png")
        rb = b.render(camera_name="default", width=32, height=24, output_path="f.png")
        assert ra["status"] == rb["status"] == "success", (ra, rb)
        assert _saved_path(ra) == str((tmp_path / "a").resolve() / "f.png")
        assert _saved_path(rb) == str((tmp_path / "b").resolve() / "f.png")
    finally:
        a.cleanup()
        b.cleanup()


def test_render_dir_is_a_known_constructor_keyword_so_a_misspelling_is_refused():
    """``own_keyword_names`` derives the accepted set from the signature, so the
    new parameter is screened without a second list to update."""
    from strands_robots.simulation.base import own_keyword_names
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    assert "render_dir" in own_keyword_names(MuJoCoSimEngine)
    with pytest.raises(TypeError, match="did you mean .render_dir"):
        MuJoCoSimEngine(rende_dir="/tmp/x")


def test_ospathlike_render_dir_is_accepted():
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    class P(os.PathLike):
        def __fspath__(self):
            return "/tmp/strands-robots-render-dir-test"

    sim = MuJoCoSimEngine(render_dir=P())
    try:
        assert sim.render_dir == Path("/tmp/strands-robots-render-dir-test").resolve()
    finally:
        sim.cleanup()
