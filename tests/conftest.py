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
from pathlib import Path
from types import ModuleType

import pytest

# Neither import below touches strands_robots, so both are safe above the
# environment defaults that the strands_robots imports further down depend on.
from tests._device_connect_real import EDGE_REBINDERS, held_modules, restore
from tests.description_clone_lock import serialize_description_clones
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
    """Register the reporters and guards this session runs with.

    :mod:`tests.session_truncation` says why a run that stops early cannot be
    read from its counts alone. :mod:`tests.description_clone_lock` says why the
    ``robot_descriptions`` clone needs a lock once the session is distributed:
    every worker collects the whole tree, so the description modules imported at
    collection time reach one shared cache directory together.
    """
    register_truncation_reporter(config)
    serialize_description_clones()


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

    Patching what is registered now is not enough on its own: a test that then
    calls :func:`tests._device_connect_real.use_the_real_edge` replaces a
    sibling's stand-in with the real edge module, and the integration it
    re-imports reads the real symbol rather than the patch. The stub is
    registered in :data:`tests._device_connect_real.EDGE_REBINDERS` as well, so
    the swap carries it onto the module the integration will read.
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

    def _rebind(module: ModuleType) -> None:
        """Bind the stub on an edge module a swap imported after this fixture ran."""
        if hasattr(module, "get_rpc_source_device"):
            monkeypatch.setattr(module, "get_rpc_source_device", _named)

    monkeypatch.setitem(EDGE_REBINDERS, "named_rpc_caller", _rebind)
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


#: Prefix of every environment variable :mod:`strands_robots.dashboard.auth`
#: reads, including the ``STORE`` that decides which file is the credential
#: record. Read by :func:`_dashboard_auth_store_is_a_per_test_file` below.
DASHBOARD_AUTH_ENV = "STRANDS_DASH_AUTH_"


@pytest.fixture(autouse=True)
def _dashboard_auth_store_is_a_per_test_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let no test read or write the credential store this machine is sealed with.

    :func:`strands_robots.dashboard.auth._store_path` resolves an unset
    ``STRANDS_DASH_AUTH_STORE`` to ``~/.strands_dashboard/auth.json`` - the file
    that holds the passkey records and the ``jwt_secret``, and so decides whether
    the dashboard on this machine is sealed. Every reader goes through
    ``_load()``, which *writes*: a machine with no store gets a fresh one
    created, and a machine whose store will not parse has the operator's file
    renamed aside and a new ``jwt_secret`` written, invalidating every live
    session token. Both under a green report, because nothing in a test asks
    where the store was.

    Sixteen modules redirected it in an autouse fixture of their own, and no two
    agreed on the rest of the family: between none and five of the sibling knobs
    unset, with the docstrings of three of them stating the rule for the other
    thirteen. Pointing it here makes the redirect a property of the session
    rather than sixteen authors remembering to, and ``tmp_path / "auth.json"`` is
    the path every one of them chose, so a module's own helper that seeds or
    reads that file needs no change.

    The rest of the family is unset for the same reason and in one sweep rather
    than from a list: a knob left set in the environment - ``RP_ID``, ``ORIGIN``,
    ``ENABLED``, ``BOOTSTRAP_TOKEN``, a duration - decides a verdict a cell is
    grading, so the developer's shell would be the thing under test. A cell that
    wants one sets it afterwards; this fixture runs first.

    The file is not created, only named: ``_save_locked`` makes the parent when
    something actually writes, so a session that never touches the dashboard
    pays a name and no I/O.
    """
    monkeypatch.setenv(DASHBOARD_AUTH_ENV + "STORE", str(tmp_path / "auth.json"))
    for name in [key for key in os.environ if key.startswith(DASHBOARD_AUTH_ENV)]:
        if name != DASHBOARD_AUTH_ENV + "STORE":
            monkeypatch.delenv(name)


#: The process-globals :mod:`strands_robots.dashboard.auth` keeps, and the value
#: each is born with. Restored by
#: :func:`_dashboard_auth_process_state_is_left_as_found` below and graded
#: against the module's own bindings in
#: ``tests/test_process_globals_do_not_cross_a_test_boundary.py``, so a fourth
#: one cannot appear there without a decision here.
DASHBOARD_AUTH_PROCESS_STATE: dict[str, object] = {"_cache": {}, "_challenges": {}, "_corrupt": None}


@pytest.fixture(autouse=True)
def _dashboard_auth_process_state_is_left_as_found() -> Iterator[None]:
    """Leave the dashboard's auth module holding nothing a test put there.

    :mod:`strands_robots.dashboard.auth` keeps three process-globals:
    ``_cache`` (the parsed credential store, keyed on the file), ``_challenges``
    (the WebAuthn ceremonies awaiting a finish, bounded by ``_CHAL_MAX`` and
    ``_CHAL_MAX_PER_IP``) and ``_corrupt`` (the diagnosis of a store that would
    not parse). All three outlive the test that filled them, so a store one test
    wrote answers a later test's read, and a ceremony one test stashed counts
    against a later test's per-ip cap.

    Measured over the fifteen ``tests/test_dashboard_auth_*`` modules run in one
    process: ``test_dashboard_auth_module`` and
    ``test_dashboard_auth_corrupt_store`` each hand the next module two pending
    challenges, and every module hands on a populated store cache.

    Fourteen of those modules reset some of it in a fixture of their own, and no
    two reset the same set - eleven reached ``_cache``, seven ``_corrupt``, three
    ``_challenges``, and one cleared the challenge table and nothing else.
    Restoring here rather than in each caller makes it a property of the session,
    and covers the names a caller left out.

    The module is looked up rather than imported so a session that never touches
    the dashboard does not pull in fastapi and webauthn. Every one of the three
    is born empty or ``None`` - unlike the predicate registry above, whose
    baseline is the shipped set - so the born values can be stated here and a
    module that first appears during a test is restored as correctly as one that
    was imported at collection.
    """
    yield
    module = sys.modules.get("strands_robots.dashboard.auth")
    if module is None:
        return
    for name, born in DASHBOARD_AUTH_PROCESS_STATE.items():
        current = getattr(module, name)
        if current == born:
            continue
        if isinstance(current, dict) and isinstance(born, dict):
            current.clear()
            current.update(born)
        else:
            setattr(module, name, born)


#: The one-shot flags :mod:`strands_robots.audit` parks on ``_AUDIT_STATE``, and
#: the value each is born with. Restored by
#: :func:`_audit_process_state_is_left_as_found` below, together with the
#: ``_SEQ_COUNTERS`` table that is born empty, and graded against the module's
#: own bindings in
#: ``tests/test_process_globals_do_not_cross_a_test_boundary.py``, so a fourth
#: flag cannot appear there without a decision here.
AUDIT_PROCESS_FLAGS: dict[str, object] = {
    "seq_loaded": False,
    "audit_log_seeded": False,
    "psk_fingerprint": None,
}


@pytest.fixture(autouse=True)
def _audit_process_state_is_left_as_found() -> Iterator[None]:
    """Leave the safety audit log's process state as the session found it.

    :mod:`strands_robots.audit` keeps four pieces of process state: the per-peer
    sequence counters (``_SEQ_COUNTERS``, born empty) and three one-shot flags on
    ``_AUDIT_STATE`` - ``seq_loaded`` and ``audit_log_seeded``, which gate the
    sidecar read and the O(records) audit-log walk to once per process, and
    ``psk_fingerprint``, the fingerprint of the PSK the first record was signed
    under. A leaked fingerprint is the sharp one: the next test to write under a
    different ``STRANDS_MESH_AUDIT_PSK`` is not signed under it, it is refused as
    a mid-run rotation and replaced by a ``PSK_DEGRADED`` poison record.

    Measured over the twenty-five modules that touch it, run in one process:
    eleven of the twenty-four module boundaries handed the next module dirty
    state - ``test_audit_serialise_safety`` and ``test_estop_lockout_race`` hand
    on all four dirty, ``test_verify_unverifiable_signed`` hands five seq
    counters to the fleet examples, and ``test_audit_log_symlink_refused`` hands
    on a PSK fingerprint.

    Every one of those modules reset some of it in a fixture of its own, and no
    two reset the same set: eleven left ``psk_fingerprint`` out, two left
    ``_SEQ_COUNTERS`` out, two left ``audit_log_seeded`` out. Restoring here
    rather than in each caller makes it a property of the session and covers the
    names a caller left out. Resets *inside* a test - the "simulate a fresh
    process" reload, the flag a test pins ``True`` to keep the log walk out of
    its subject - stay where they are; they are the test's own subject, not
    isolation.

    The module is looked up rather than imported so a session that never writes a
    safety event does not pull it in. All four are born empty or ``False`` or
    ``None``, so the born values can be stated here and a module first imported
    during a test is restored as correctly as one imported at collection.
    """
    yield
    module = sys.modules.get("strands_robots.audit")
    if module is None:
        return
    if module._SEQ_COUNTERS:
        module._SEQ_COUNTERS.clear()
    for name, born in AUDIT_PROCESS_FLAGS.items():
        if getattr(module._AUDIT_STATE, name) != born:
            setattr(module._AUDIT_STATE, name, born)


#: Every "warn once per process" memo in the package, as ``(module, name)``: a
#: module-level container of the keys already reported, born empty, whose only
#: job is to stop a repeat. Sixteen of them across eleven modules. Emptied by
#: :func:`_warn_once_memos_are_left_empty` below, and discovered rather than
#: trusted in ``tests/test_process_globals_do_not_cross_a_test_boundary.py``, so
#: a seventeenth cannot appear without a row here.
WARN_ONCE_MEMOS: tuple[tuple[str, str], ...] = (
    ("strands_robots._mesh_switch", "_UNKNOWN_WARNED"),
    ("strands_robots.device_connect._authz", "_warned_permissive"),
    ("strands_robots.device_connect._authz", "_warned_insecure_acl"),
    ("strands_robots.device_connect._authz", "_warned_unconfigured"),
    ("strands_robots.mesh._backend_select", "_UNKNOWN_WARNED"),
    ("strands_robots.mesh._zenoh_config", "_NON_POSIX_TLS_WARNED_KEYS"),
    ("strands_robots.mesh.core", "_POSTURE_WARNINGS_EMITTED"),
    ("strands_robots.mesh.iot.provision", "_UNVERIFIED_CA_WARNED"),
    ("strands_robots.mesh.session", "_RETENTION_WARNED"),
    ("strands_robots.mesh.session", "_unencodable_topics_warned"),
    ("strands_robots.mesh.session", "_zenoh_missing_warned"),
    ("strands_robots.mesh.transport.bridge_transport", "_WARNED_HALF_BRIDGED_HEADS"),
    ("strands_robots.policies.lerobot_local.embodiment", "_WARNED_STATE_KEY_MISMATCH"),
    ("strands_robots.simulation.mujoco.backend", "_software_render_warned"),
    ("strands_robots.simulation.predicates", "_RESOLUTION_WARNED"),
    ("strands_robots.simulation.predicates", "_WARNED_NO_CONTACT_QUERY"),
)


@pytest.fixture(autouse=True)
def _warn_once_memos_are_left_empty() -> Iterator[None]:
    """Leave no key spent in a warn-once memo once a test is over.

    Sixteen places in the package report a posture once per process and keep the
    keys they have reported in a module-level set - an unrecognised env value, a
    TLS key with non-POSIX permissions, a state key a checkpoint does not
    declare, a body a predicate spec cannot resolve. The dedup is deliberate:
    the second report would be noise, and in the reward/eval hot loop it would
    be a flood. It also means the *first* test to spend a key decides that every
    later test reading that report gets silence - and the keys are shared, which
    is why one of the resets this replaces carried the comment "so this
    assertion is independent of what other predicate tests warned first (the
    'robot base' key is shared)".

    Measured over the twenty-three modules that touch the mesh ones, run in one
    process: ``_POSTURE_WARNINGS_EMITTED`` was dirty at the end of 602 of 616
    tests and at all 22 module boundaries, ``_zenoh_missing_warned`` at 544 and
    20. Nobody reset either. Thirty-three modules reset one at sixty-one sites -
    each only the memo it had been bitten by, in its own fixture with its own
    docstring for one behaviour - and two of them *restored* the keys they found
    spent, handing the next module exactly the leak they had cleared for
    themselves.

    Emptying them here makes it a property of the session, so a test that asserts
    a report can assume it is the first to ask. A reset *inside* a test - between
    two asks that grade the once-per-process gate itself - is that test's own
    subject and stays where it is.

    Each module is looked up rather than imported, so a session that never
    touches the mesh or a policy does not pull one in, and every memo is born
    empty, so emptying is the whole restore.
    """
    yield
    for module_name, name in WARN_ONCE_MEMOS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        memo = getattr(module, name, None)
        if memo:
            memo.clear()
