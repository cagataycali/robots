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

Finally, registers the session-truncation reporter from
:mod:`tests.session_truncation`, so a run that stops before every collected test
has started says so instead of reporting counts that read as a total.
"""

import os
import sys

import pytest

# Neither import below touches strands_robots, so both are safe above the
# environment defaults that the strands_robots imports further down depend on.
from tests.session_truncation import register_truncation_reporter

# Disable mesh BEFORE any strands_robots import below pulls in robot.py.
# Use setdefault so tests that explicitly enable the mesh (e.g. integ tests)
# can override via the environment without conftest stomping on them.
os.environ.setdefault("STRANDS_MESH", "false")

# Disable the Device Connect dispatch path in robot_mesh by default so unit
# tests exercise the built-in mesh deterministically, without opening real
# Device Connect (Zenoh) connections. The GUIDE E2E demo runs outside pytest
# and leaves this unset, so Device Connect remains the primary path at runtime.
os.environ.setdefault("STRANDS_ROBOT_MESH_DC", "off")

# Choose MuJoCo's GL backend once for the whole session, before any test module
# is imported. 29 modules under tests/ set it at import time with setdefault,
# and 18 of those hard-coded "egl", which mujoco refuses on macOS ("invalid
# value for environment variable MUJOCO_GL: egl"). Collection imports every
# module, so whichever of them pytest reached first decided the value for the
# whole session. Set here, a module-level setdefault under tests/ is a no-op
# whatever it says, and a user's own MUJOCO_GL still wins. tests_integ/ has no
# conftest, so the defaults in its own modules stay load-bearing.
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from tests.mocks.torch_mock import install_torch_mock

# Must run before any test imports policy modules
install_torch_mock()


def pytest_configure(config: pytest.Config) -> None:
    """Report the size of a session that stops before every test has started.

    See :mod:`tests.session_truncation` for why the counts alone do not say it.
    """
    register_truncation_reporter(config)


@pytest.fixture
def named_rpc_caller(monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the test as an allowlisted Device Connect operator.

    ``is_authorized_caller`` fails CLOSED: with ``DEVICE_CONNECT_RPC_ALLOW``
    unset nobody may call an RPC, and even ``*`` admits only a NAMED caller.
    Tests that grade what an RPC does once admitted (delegation, argument
    domains, stop reporting) opt into this fixture with a module-level
    ``pytestmark = pytest.mark.usefixtures("named_rpc_caller")``; the
    authorization decision itself is graded in
    ``tests/test_device_connect_hardening.py``, which never uses it.

    The ``@rpc`` wrapper sets the caller ContextVar from the ``source_device``
    kwarg the runtime injects - ``None`` when a test awaits the handler
    directly - so setting the variable here would be undone on every call.
    ``get_rpc_source_device`` is patched instead, both where the drivers
    import it from (so a driver module re-imported after this fixture binds
    the stub) and on every driver module already imported. A module that has
    replaced ``device_connect_edge`` with a mock gives that mock's
    ``get_rpc_source_device`` the same name itself. A test that sets its own
    allowlist or patches the symbol again still wins: both apply after this.
    """
    import importlib
    import sys

    caller = "test-operator"
    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", caller)
    # emergencyStop inherits the RPC allowlist when it has none of its own, and
    # the stop tests here arrive from other named devices ("other-robot"); any
    # named caller may stop. A test that grades an ignored stop sets its own.
    monkeypatch.setenv("DEVICE_CONNECT_ESTOP_ALLOW", "*")

    def _named() -> str:
        return caller

    # A driver module (re)imported during the test binds the symbol from
    # ``device_connect_edge.drivers``, so the real package is brought in now
    # (unless a sibling module has a mock in its place, which then answers
    # itself) and patched before that import can happen.
    for name in ("device_connect_edge.drivers", "device_connect_edge.drivers.decorators"):
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:  # a mocked parent package is not importable through
                continue
        if hasattr(module, "__file__") and hasattr(module, "get_rpc_source_device"):
            monkeypatch.setattr(module, "get_rpc_source_device", _named)
    for name, module in list(sys.modules.items()):
        if name.startswith("strands_robots.device_connect") and hasattr(module, "get_rpc_source_device"):
            monkeypatch.setattr(module, "get_rpc_source_device", _named)
    return caller
