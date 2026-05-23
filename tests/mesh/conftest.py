"""Shared fixtures for the mesh test suite.

Default ``STRANDS_MESH_AUTH_MODE`` to ``none`` so tests that mock Zenoh
do not have to provide cert files. Tests that exercise the mTLS code
path opt in by setting the env var explicitly via ``monkeypatch``.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _default_mesh_auth_mode_none(monkeypatch):
    """Default to auth_mode=none for mesh unit tests.

    Production deployments default to mTLS (the value of
    ``STRANDS_MESH_AUTH_MODE`` when no env var is set). The test suite
    needs the opposite: most tests use a mocked Zenoh session and do
    not have CA / cert / key files to point at.
    """
    if "STRANDS_MESH_AUTH_MODE" not in os.environ:
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
