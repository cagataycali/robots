"""Shared test fixtures and configuration.

Installs a numpy-backed torch stand-in when real torch is unavailable, so the
parts of the suite that need only a thin tensor surface run without the ~2GB
dependency. That stand-in is a subset rather than a replacement: a test reaching
outside it is skipped with the attribute and the remedy named, not failed.

Also disables the Zenoh mesh by default during the test suite so the
``Robot()`` / ``Simulation()`` factory does not spin up real Zenoh
sessions and background heartbeat threads when ``eclipse-zenoh`` is
installed in the test environment.  Mesh-specific tests opt back in
explicitly via ``monkeypatch.delenv`` or by patching ``init_mesh``.
"""

import os

# Disable mesh BEFORE any strands_robots import below pulls in robot.py.
# Use setdefault so tests that explicitly enable the mesh (e.g. integ tests)
# can override via the environment without conftest stomping on them.
os.environ.setdefault("STRANDS_MESH", "false")

# Disable the Device Connect dispatch path in robot_mesh by default so unit
# tests exercise the built-in mesh deterministically, without opening real
# Device Connect (Zenoh) connections. The GUIDE E2E demo runs outside pytest
# and leaves this unset, so Device Connect remains the primary path at runtime.
os.environ.setdefault("STRANDS_ROBOT_MESH_DC", "off")

from tests.mocks.torch_mock import install_torch_mock

# Must run before any test imports policy modules
install_torch_mock()
