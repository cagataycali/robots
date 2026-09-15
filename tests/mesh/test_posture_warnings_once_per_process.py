"""The startup posture warnings are said once per process, not once per Mesh.

Observed: ``examples/04_mesh_peer_discovery.py`` printed the four-line
``No emergency-stop resume code set`` banner twice - once as
``[safety:example-arm-01]`` for the sim, once as
``[safety:example-arm-01__so100]`` for its per-robot child peer. Both read the
same ``STRANDS_MESH_OVERRIDE_CODE``; the second banner carried no new fact.
The multicast warning has the same shape. Same harness as
``test_multicast_startup_warning``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh import core as mesh_core

_OVERRIDE_MARKER = "No emergency-stop resume code set"
_MULTICAST_MARKER = "Multicast scouting is ON"


class _StubDecl:
    def undeclare(self) -> None:
        pass


class _StubSession:
    def declare_subscriber(self, *args, **kwargs):
        return _StubDecl()


def _make_mesh(peer_id: str) -> Mesh:
    robot = SimpleNamespace(
        tool_name_str=peer_id,
        robot=SimpleNamespace(
            is_connected=True,
            name=f"{peer_id}_test",
            config=SimpleNamespace(cameras={}),
            get_observation=MagicMock(return_value={}),
        ),
    )
    return Mesh(robot, peer_id=peer_id, peer_type="robot")


def _run_start(mesh: Mesh, caplog: pytest.LogCaptureFixture) -> None:
    with patch.object(mesh_core, "get_session", return_value=_StubSession()):
        with patch.object(mesh_core, "release_session"):
            with patch.object(mesh, "_heartbeat_loop"), patch.object(mesh, "_state_loop"):
                with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
                    mesh.start()
                mesh.stop()


@pytest.fixture(autouse=True)
def _fresh_process_state():
    mesh_core._reset_posture_warnings()
    yield
    mesh_core._reset_posture_warnings()


def _messages(caplog: pytest.LogCaptureFixture, marker: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if marker in r.getMessage()]


def test_two_meshes_in_one_process_say_the_override_banner_once(monkeypatch, caplog):
    monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)
    monkeypatch.setenv("STRANDS_MESH_MULTICAST", "false")
    _run_start(_make_mesh("parent-sim"), caplog)
    _run_start(_make_mesh("parent-sim__so100"), caplog)
    banners = _messages(caplog, _OVERRIDE_MARKER)
    assert len(banners) == 1, banners
    assert "[safety:parent-sim]" in banners[0], "the first peer to start names the banner"


def test_the_first_mesh_still_warns(monkeypatch, caplog):
    monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)
    monkeypatch.setenv("STRANDS_MESH_MULTICAST", "false")
    _run_start(_make_mesh("solo"), caplog)
    assert len(_messages(caplog, _OVERRIDE_MARKER)) == 1


def test_a_set_code_keeps_the_banner_silent(monkeypatch, caplog):
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "1234")
    monkeypatch.setenv("STRANDS_MESH_MULTICAST", "false")
    _run_start(_make_mesh("coded"), caplog)
    assert _messages(caplog, _OVERRIDE_MARKER) == []


def test_multicast_banner_is_also_once_per_process(monkeypatch, caplog):
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "1234")
    monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
    _run_start(_make_mesh("mc-a"), caplog)
    _run_start(_make_mesh("mc-b"), caplog)
    assert len(_messages(caplog, _MULTICAST_MARKER)) == 1


def test_the_two_kinds_are_independent(monkeypatch, caplog):
    monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)
    monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
    _run_start(_make_mesh("both"), caplog)
    assert len(_messages(caplog, _OVERRIDE_MARKER)) == 1
    assert len(_messages(caplog, _MULTICAST_MARKER)) == 1
