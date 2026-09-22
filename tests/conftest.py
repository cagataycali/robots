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
from collections.abc import Iterator

import pytest

# Neither import below touches strands_robots, so both are safe above the
# environment defaults that the strands_robots imports further down depend on.
from tests._device_connect_real import held_modules, restore
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


@pytest.fixture(autouse=True)
def _device_connect_modules_are_put_back() -> Iterator[None]:
    """Undo any swap of the Device Connect integration this test performed.

    Thirteen test modules run against the real ``device_connect_edge`` by
    dropping ``strands_robots.device_connect.*`` from ``sys.modules`` so the
    integration re-imports against the genuine ``@rpc`` / ``DeviceDriver``
    (:func:`tests._device_connect_real.use_the_real_edge`). Dropping an entry is
    not an undo: every reference a sibling module bound at collection time is
    orphaned, and the next import hands out a different object - so a
    ``monkeypatch.setattr`` on the sibling's binding lands on a module the code
    under test no longer reads.

    Measured, with ``tests/test_device_connect_hardening.py`` running ahead of
    the reachy driver files (the ordering ``-p xdist --dist loadfile`` produces
    and a serial run does not): four cells in
    ``tests/drivers/test_reachy_wireless_daemon_protocol.py`` resolved
    ``reachy-a.local`` for real and failed. Restoring here rather than in each
    caller keeps the pair together - the swap is undone by the session, not by
    thirteen callers remembering to.
    """
    held = held_modules()
    try:
        yield
    finally:
        if held_modules() != held:
            restore(held)


@pytest.fixture(autouse=True)
def _mesh_rate_limit_history_is_left_empty() -> Iterator[None]:
    """Leave no rate-limit slots consumed once a test is over.

    ``strands_robots.tools.robot_mesh`` bounds LLM-driven nuisance with a
    process-global sliding window (``_RATE_HISTORY``, 30 ``tell`` calls per
    60 s). Every accepted tool call consumes a slot for the life of the
    process, so a test that spends the window makes the *next* test's call
    return "rate limit exceeded" instead of doing the thing it asserts.

    Measured with ``tests/test_hitl_operator_response_audit.py`` running ahead
    of ``tests/mesh/test_robot_mesh_tool.py`` (the ordering ``--dist loadfile``
    produces and a serial run does not): that file drains ``tell`` to exactly
    its limit of 30 to make the post-approval re-check deterministic, and four
    cells in the victim then failed on the refusal - one reading ``'error' ==
    'success'``, two on "rate limit exceeded" where a dispatch error was
    expected, one on a call that never reached the mesh at all.

    Ten test modules used to reset the window in a fixture of their own,
    each with a docstring saying the cases must stay independent of collection
    order. Clearing here rather than in each caller makes that a property of
    the session: the window a test spends is refunded by the session, not by
    ten callers remembering to. Resets *inside* a test - a case that needs two
    accepted calls of one action - stay where they are; they are the test's
    own subject, not isolation.

    The module is looked up rather than imported so a session that never
    touches the mesh does not pull it in.
    """
    yield
    module = sys.modules.get("strands_robots.tools.robot_mesh")
    if module is not None:
        module._reset_rate_limits()


@pytest.fixture(autouse=True)
def _optional_module_memo_holds_no_stand_in() -> Iterator[None]:
    """Leave no stand-in module memoised once a test is over.

    ``strands_robots.utils.require_optional`` memoises every optional
    dependency it resolves in a process-global dict (``_lazy_modules``), and a
    test stands in for a module nothing installs by rebinding ``sys.modules``.
    ``monkeypatch.setitem`` restores the binding, but the memo is a second one
    it cannot reach: the stand-in the package cached during the test is then
    handed to every later caller in the process, whose production code calls a
    fake the fixture already took away.

    Measured with ``tests/policies/moveit2/test_zmq_sidecar.py`` running ahead
    of the groot client files (the ordering ``--dist loadfile`` produces and a
    serial run does not): that file's ZMQ stand-in carries
    ``Context = SimpleNamespace(instance=...)``, the memo kept it, and 42 cells
    across three files died in ``Gr00tInferenceClient.__init__`` /
    ``MoveIt2Client`` on ``TypeError: 'types.SimpleNamespace' object is not
    callable``.

    Restoring here rather than in each caller makes it a property of the
    session: the memo a test fills is emptied by the session, not by every
    author of a stand-in remembering to. Entries a test installs *itself*
    (``monkeypatch.setitem(utils._lazy_modules, ...)`` - the seam that injects a
    fake into the package directly) are monkeypatch's to undo and are left
    alone; this restores what the package cached on its own behalf.

    The module is looked up rather than imported so a session that never
    touches it does not pull it in.
    """
    memo = getattr(sys.modules.get("strands_robots.utils"), "_lazy_modules", None)
    before = dict(memo) if memo is not None else {}
    yield
    memo = getattr(sys.modules.get("strands_robots.utils"), "_lazy_modules", None)
    if memo is not None and memo != before:
        memo.clear()
        memo.update(before)


@pytest.fixture(autouse=True)
def _predicate_registry_is_left_as_found() -> Iterator[None]:
    """Leave the predicate registry holding only what the session started with.

    ``strands_robots.simulation.predicates.PREDICATE_REGISTRY`` is a
    process-global dict, and :func:`register_predicate` is the documented way
    to extend it. A test that registers one leaves it there for every later
    test in the process, and a grader that reads the registry as the set of
    shipped predicates then fails on a name that only a test knows.

    Measured with ``tests/test_fleet_emergency_evacuation.py`` running ahead of
    ``tests/simulation/test_predicate_docstring_completeness.py`` (the ordering
    ``--dist loadfile`` produces and a serial run does not): the example under
    test registers ``evacuation_abort_within``, and the docstring grader read it
    as drift - ``bool docstring drift: missing=['evacuation_abort_within']``.

    Seven call sites used to undo their own registration in a ``try``/
    ``finally``; the session owns it now, so a registration is one line again
    and the one path that forgot is covered too.

    The module is imported here rather than looked up in ``sys.modules`` the way
    the ``_lazy_modules`` memo above is: that memo is born empty, this registry
    is born holding the 30 shipped predicates. A lookup that misses the module -
    which is what happens whenever nothing imported it at collection time, as in
    ``pytest tests/test_fleet_emergency_evacuation.py`` alone, where the example
    under test imports it inside a test - would take an empty baseline and this
    teardown would then wipe the shipped set for the rest of the process, leaving
    every later cell on ``Unknown predicate 'inside_region'``. The import costs
    0.1 s once and pulls in stdlib plus :mod:`strands_robots.utils` only.
    """
    from strands_robots.simulation import predicates

    before = dict(predicates.PREDICATE_REGISTRY)
    yield
    if predicates.PREDICATE_REGISTRY != before:
        predicates.PREDICATE_REGISTRY.clear()
        predicates.PREDICATE_REGISTRY.update(before)
