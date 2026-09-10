"""The WIRE SECURITY DISABLED line must name the variable the operator set.

``STRANDS_MESH_LOCAL_DEV`` alone turns wire auth off; the line used to claim
the operator "opted in via STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1" -- a
variable they never set -- so the log pointed a reader at the wrong knob.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.mesh import _acl_config
from strands_robots.mesh.session import _build_config


@pytest.fixture(autouse=True)
def _no_stale_thread_snapshot():
    """``_build_config`` prefers a thread-local ``auth_mode`` stashed by an
    earlier ``Mesh.start`` over the environment; a test that left one behind
    would make these read ``mtls`` instead of the knobs they set."""
    _acl_config._clear_thread_snapshot()
    yield
    _acl_config._clear_thread_snapshot()


def _disabled_line(caplog: pytest.LogCaptureFixture) -> str:
    lines = [r.getMessage() for r in caplog.records if "WIRE SECURITY DISABLED" in r.getMessage()]
    assert len(lines) == 1, lines
    return lines[0]


def test_local_dev_is_named_when_it_opened_the_gate(monkeypatch, caplog):
    monkeypatch.delenv("STRANDS_MESH_AUTH_MODE", raising=False)
    monkeypatch.delenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", raising=False)
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "true")
    with caplog.at_level(logging.ERROR):
        _build_config()
    line = _disabled_line(caplog)
    assert "opted in via STRANDS_MESH_LOCAL_DEV" in line
    assert "I_KNOW_THIS_IS_INSECURE" not in line


def test_explicit_second_factor_is_named_when_it_opened_the_gate(monkeypatch, caplog):
    monkeypatch.delenv("STRANDS_MESH_LOCAL_DEV", raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
    monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")
    with caplog.at_level(logging.ERROR):
        _build_config()
    assert "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1" in _disabled_line(caplog)
