"""Shared fixtures for the mesh test suite.

Default ``STRANDS_MESH_AUTH_MODE`` to ``none`` so tests that mock Zenoh
do not have to provide cert files. Tests that exercise the mTLS code
path opt in by setting the env var explicitly via ``monkeypatch``.

Tests that exercise the second-factor opt-in gate itself
(``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE``) MUST opt out of the autouse
default by adding the ``mesh_auth_mode_default`` marker, otherwise the
gate is short-circuited by this fixture and the test passes by
accident::

    @pytest.mark.mesh_auth_mode_default
    def test_second_factor_required():
        # neither STRANDS_MESH_AUTH_MODE nor _I_KNOW_THIS_IS_INSECURE
        # are set by the autouse fixture for this test.
        ...
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register markers used by mesh tests."""
    config.addinivalue_line(
        "markers",
        "mesh_auth_mode_default: opt out of the autouse "
        "STRANDS_MESH_AUTH_MODE=none default. Use for tests that "
        "exercise the second-factor opt-in gate itself.",
    )


@pytest.fixture(autouse=True)
def _default_mesh_auth_mode_none(monkeypatch, request):
    """Default to auth_mode=none for mesh unit tests.

    Production deployments default to mTLS (the value of
    ``STRANDS_MESH_AUTH_MODE`` when no env var is set). The test suite
    needs the opposite: most tests use a mocked Zenoh session and do
    not have CA / cert / key files to point at.

    The second-factor opt-in flag (``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE``)
    is set in the *same conditional branch* as the auth-mode override so
    that a developer who runs ``STRANDS_MESH_AUTH_MODE=mtls hatch run test``
    does not get the insecure-opt-in flag silently injected.

    Tests that exercise the second-factor gate itself MUST opt out via
    ``@pytest.mark.mesh_auth_mode_default``, otherwise the gate passes
    by accident under this fixture.
    """
    if request.node.get_closest_marker("mesh_auth_mode_default"):
        return
    if "STRANDS_MESH_AUTH_MODE" not in os.environ:
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        # B2: auth_mode=none requires explicit second-factor opt-in.
        monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")
