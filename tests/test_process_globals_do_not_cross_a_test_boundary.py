"""Every process-global a test dirties is as the session found it by the next test.

Six kinds of binding in this package outlive the test that fills them - the mesh
rate-limit window, the optional-dependency memo, the predicate registry, the
dashboard's auth state, the safety audit log's sequence counters and one-shot
flags, and the sixteen warn-once memos that remember which postures have already
been reported - so the test that fills one decides what a later test reads. ``tests/conftest.py`` restores each in an autouse fixture, and the pin is
one shape: dirty it in one cell, find it as found in the next, with no fixture in
this file. Three files each pinned one global in that shape; the shape is a table
now, one row per global, so a global that joins the session's set joins the pin
with a row.

The two parametrised cells run in this order, which is the order pytest runs
them, and the first also asserts what it was handed - so every row grades the
restore that happened at the previous boundary as well as its own.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import strands_robots
import strands_robots.audit as audit
import strands_robots.dashboard.auth as auth
import strands_robots.tools.robot_mesh as rmt
from strands_robots.simulation.predicates import PREDICATE_REGISTRY, register_predicate
from strands_robots.utils import require_optional
from tests.conftest import AUDIT_PROCESS_FLAGS, DASHBOARD_AUTH_PROCESS_STATE, WARN_ONCE_MEMOS

#: The mesh action whose window a row spends. Any of them would do; this one is
#: the rate-limited verb the tool's own tests drain.
MESH_ACTION = "tell"

#: A dependency nothing installs, so the memo can only hold a test's stand-in.
ABSENT_DEPENDENCY = "strands_robots_absent_optional_dependency"

#: A predicate name only this module knows, so the registry row cannot collide
#: with a shipped one.
PREDICATE_NAME = "a_predicate_only_this_module_knows"

#: The key the warn-once row spends. Nothing in the package reports this posture,
#: so a memo holding it can only have been filled by that row.
WARN_ONCE_KEY = "a posture only this module reported"


@dataclass(frozen=True)
class ProcessGlobal:
    """One process-global the session restores, and how to see that it did.

    :param label: How the row reads in a test id.
    :param dirty: Fill it the way a test legitimately would - the documented
        extension point, the tool call, the stand-in - never by reaching past the
        production seam, so a row also pins that the legitimate way is reachable.
    :param is_as_found: ``True`` while it holds what the session started with.
    """

    label: str
    dirty: Callable[[pytest.MonkeyPatch], None]
    is_as_found: Callable[[], bool]


def _spend_the_mesh_window(_monkeypatch: pytest.MonkeyPatch) -> None:
    """Draining the window is legitimate - several tests do it deliberately."""
    limit, _window = rmt._RATE_LIMITS[MESH_ACTION]
    for _ in range(limit):
        assert rmt._rate_limit_check_and_record(MESH_ACTION) is None


def _memoise_a_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standing in through ``sys.modules`` is legitimate - many tests do it.

    ``monkeypatch`` restores the binding it was given, not the copy
    ``require_optional`` memoised, which is the whole reason the session has to.
    """
    stand_in = types.ModuleType(ABSENT_DEPENDENCY)
    monkeypatch.setitem(sys.modules, ABSENT_DEPENDENCY, stand_in)
    assert require_optional(ABSENT_DEPENDENCY) is stand_in


def _memo_is_as_found() -> bool:
    try:
        require_optional(ABSENT_DEPENDENCY)
    except ImportError:
        return True
    return False


def _register_a_predicate(_monkeypatch: pytest.MonkeyPatch) -> None:
    """Registering is legitimate - it is the documented extension point."""
    register_predicate(PREDICATE_NAME, lambda: lambda _sim: True)


def _fill_the_dashboard_auth_state(_monkeypatch: pytest.MonkeyPatch) -> None:
    """Fill all three at once: a caller that reset one of them left two behind.

    Written onto the globals rather than through ``_load`` / ``_stash_challenge``
    so the row needs no credential store of its own - what is pinned is the
    restore, not how the state got there.
    """
    auth._cache[("a store only this test read",)] = auth._CachedStore(raw="{}", store={})
    auth._challenges["a ceremony only this test began"] = {"t_mono": time.monotonic()}
    auth._corrupt = {"path": "a diagnosis only this test made"}


def _dashboard_auth_state_is_as_found() -> bool:
    return all(getattr(auth, name) == born for name, born in DASHBOARD_AUTH_PROCESS_STATE.items())


def _fill_the_audit_process_state(_monkeypatch: pytest.MonkeyPatch) -> None:
    """Fill all four at once: a caller that reset three of them left one behind.

    Written onto the globals rather than through :func:`audit.log_safety_event`
    so the row needs no audit directory of its own - what is pinned is the
    restore, not how the state got there. ``audit_log_seeded`` has no cheaper
    fill in any case: the only path that sets it is the audit-log walk a
    corrupt sidecar falls back to.
    """
    audit._SEQ_COUNTERS["a peer only this test wrote as"] = 1
    audit._AUDIT_STATE.seq_loaded = True
    audit._AUDIT_STATE.audit_log_seeded = True
    audit._AUDIT_STATE.psk_fingerprint = b"a fingerprint only this test saw"


def _audit_process_state_is_as_found() -> bool:
    return not audit._SEQ_COUNTERS and all(
        getattr(audit._AUDIT_STATE, name) == born for name, born in AUDIT_PROCESS_FLAGS.items()
    )


def _spend_every_warn_once_key(_monkeypatch: pytest.MonkeyPatch) -> None:
    """Spend the same key in all sixteen: a caller that emptied one left fifteen.

    Written onto the memos rather than by provoking sixteen reports - each of
    which needs its own environment, backend or checkpoint - because what is
    pinned is the restore, not how the key got there. The modules are imported
    here so the row grades all sixteen whatever else the session touched.
    """
    for memo in _warn_once_memos():
        if isinstance(memo, set):
            memo.add(WARN_ONCE_KEY)
        else:
            memo[WARN_ONCE_KEY] = None


def _warn_once_memos() -> list[Any]:
    """The live container behind every roster row, importing what is absent."""
    memos = []
    for module_name, name in WARN_ONCE_MEMOS:
        memos.append(getattr(importlib.import_module(module_name), name))
    return memos


def _warn_once_memos_are_as_found() -> bool:
    return not any(memo for memo in _warn_once_memos())


PROCESS_GLOBALS = [
    ProcessGlobal(
        "the mesh rate-limit window", _spend_the_mesh_window, lambda: rmt._rate_limit_check(MESH_ACTION) is None
    ),
    ProcessGlobal("the optional-dependency memo", _memoise_a_stand_in, _memo_is_as_found),
    ProcessGlobal("the predicate registry", _register_a_predicate, lambda: PREDICATE_NAME not in PREDICATE_REGISTRY),
    ProcessGlobal("the dashboard auth state", _fill_the_dashboard_auth_state, _dashboard_auth_state_is_as_found),
    ProcessGlobal("the audit process state", _fill_the_audit_process_state, _audit_process_state_is_as_found),
    ProcessGlobal("the warn-once memos", _spend_every_warn_once_key, _warn_once_memos_are_as_found),
]

_IDS = [state.label for state in PROCESS_GLOBALS]


@pytest.mark.parametrize("state", PROCESS_GLOBALS, ids=_IDS)
def test_a_test_may_dirty_a_process_global(state: ProcessGlobal, monkeypatch: pytest.MonkeyPatch) -> None:
    """Filling it is legitimate, and this cell was handed it clean to begin with."""
    assert state.is_as_found(), f"the session handed this cell a dirty {state.label}"
    state.dirty(monkeypatch)
    assert not state.is_as_found(), f"this row's dirty() did not fill {state.label}"


@pytest.mark.parametrize("state", PROCESS_GLOBALS, ids=_IDS)
def test_the_next_test_finds_the_process_global_as_found(state: ProcessGlobal) -> None:
    """The restore happened at the boundary, with no fixture in this file."""
    assert state.is_as_found(), f"{state.label} carried a previous test's value"


def test_the_dashboard_auth_roster_names_every_global_that_module_keeps() -> None:
    """A fourth global in that module joins the roster, or this cell says so.

    The roster in ``tests/conftest.py`` is what the session restores, and a
    global added beside these three would be restored by nobody - which is how
    fourteen callers came to reset three different subsets. Discovered rather
    than trusted: a module-level binding whose name is lower-case after the
    leading underscore, and which is neither callable, a module, nor a lock.
    """
    locks = (type(threading.Lock()), type(threading.RLock()))
    discovered = {
        name
        for name, value in vars(auth).items()
        if name.startswith("_")
        and not name.startswith("__")
        and not name[1:].isupper()
        and not callable(value)
        and not isinstance(value, (types.ModuleType, *locks))
    }
    assert discovered == set(DASHBOARD_AUTH_PROCESS_STATE)


def test_the_audit_roster_names_every_piece_of_state_that_module_keeps() -> None:
    """A fourth flag, or a second table, joins the roster - or this cell says so.

    Two things can grow: the flags parked on ``_ProcessAuditState``, whose
    ``__slots__`` is the module's own list of them, and the mutable module-level
    bindings themselves. A flag added to that class, or a second per-peer table
    beside ``_SEQ_COUNTERS``, would be restored by nobody - which is how
    twenty-five callers came to reset four different subsets. Discovered rather
    than trusted: a module-level binding that is a dict, or whose type is defined
    in the module, is mutable state; the rest of the private names there are
    ``int`` / ``str`` / ``Path`` constants and locks.
    """
    assert set(audit._ProcessAuditState.__slots__) == set(AUDIT_PROCESS_FLAGS)
    mutable = {
        name
        for name, value in vars(audit).items()
        if name.startswith("_")
        and not name.startswith("__")
        and (isinstance(value, dict) or type(value).__module__ == audit.__name__)
    }
    assert mutable == {"_SEQ_COUNTERS", "_AUDIT_STATE"}


def test_the_warn_once_roster_names_every_memo_in_the_package() -> None:
    """A seventeenth warn-once memo joins the roster - or this cell says so.

    The memo is the whole implementation of "report this once per process", so a
    new one is added wherever a new report is added, by someone who is not
    thinking about test isolation - and it would be emptied by nobody, which is
    how thirty-three modules came to reset one memo each. Discovered from the
    source rather than from imported modules, so a module this session never
    imports is still graded: a module-level binding whose name says ``warn`` and
    whose value is a fresh ``set()`` or ``OrderedDict()`` is a memo, and the
    locks beside them (``_POSTURE_WARNINGS_LOCK``,
    ``_NON_POSIX_TLS_WARNED_LOCK``) are not.
    """
    package = Path(strands_robots.__file__).parent
    discovered = set()
    for path in sorted(package.rglob("*.py")):
        module = ".".join(("strands_robots", *path.relative_to(package).with_suffix("").parts))
        module = module.removesuffix(".__init__")
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                target, value = node.targets[0].id, node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target, value = node.target.id, node.value
            else:
                continue
            if value is None or "warn" not in target.lower():
                continue
            if ast.unparse(value) in ("set()", "OrderedDict()", "dict()", "{}"):
                discovered.add((module, target))
    assert discovered == set(WARN_ONCE_MEMOS)


def test_a_session_that_first_imports_the_registry_inside_a_test_keeps_the_shipped_set() -> None:
    """The predicate baseline is the shipped set even when collection never imported it.

    The row above cannot see this: ``tests/test_fleet_emergency_evacuation.py``
    imports the predicates module for the first time *inside* a test - its
    example registers ``evacuation_abort_within`` lazily - so a teardown that
    reads ``sys.modules`` at setup finds no module, takes an empty baseline and
    wipes the 30 shipped predicates for the rest of the process: every later cell
    dies on ``ValueError: Unknown predicate 'inside_region'``. Run in a
    subprocess because any session that collects *this* file has already imported
    the module, which is exactly what masks the empty-baseline path.
    """
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_fleet_emergency_evacuation.py",
            "--no-cov",
            "-p",
            "no:randomly",
            "-q",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
