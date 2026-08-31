"""Every DDS endpoint in the process is constructed under the one shared lock.

``_g1_common``'s module docstring states the contract this file grades: "A
``ChannelSubscriber`` and a ``ChannelPublisher`` cannot be constructed
concurrently: the CycloneDDS bindings segfault. ``_DDS_INIT_LOCK`` is the
*shared* lock the driver and the tools (issue #358) both hold while creating
readers or writers. One lock; two consumers." ``_dds_engine.subscribe``
repeats it verbatim.

A lock like that is only a guarantee where every caller takes it, and the
caller that loses the race is the one holding nothing - so the unconverted
side is the side that breaks. That is why this is a rule over the source and
not a test per call site: the failure mode is a *new* endpoint-creating call
site that takes a different lock, or no lock, and a per-site test cannot fail
for a site nobody wrote yet. ``use_unitree._get_client`` was exactly that
site. It held only the module-private ``_CLIENTS_LOCK`` while running
``cls()`` / ``SetTimeout()`` / ``Init()``, and ``Init()`` is what builds an
RPC client's DDS request/response endpoints - so an agent thread's first call
to any service could construct endpoints while the driver's engine
constructed subscribers on its own threads (streaming, policy rollout, mesh
telemetry). A second lock nearby is not the same guarantee; the two locks do
not know about each other.

The cost of losing is why this cannot be graded by "it returns an error
envelope": a segfault in a native binding is not a Python exception, so the
dispatcher's "return an envelope, never raise" boundary cannot catch it. The
process dies, possibly mid-motion on a live robot.

**The rule derives its own subject.** The set of endpoint-creating operations
is read out of the modules that *own* the contract - the private
infrastructure modules under ``tools/g1`` (``_``-prefixed, and containing a
``with _DDS_INIT_LOCK`` block), which are where the lock is defined and where
the package decides what "creating an endpoint" means. An operation counts as
endpoint-creating when those modules only ever perform it under the lock.
Deriving from the infrastructure layer rather than package-wide is
deliberate: were the set derived from every module, the very violation this
file exists to catch would remove its own operation from the set and the rule
would pass on the broken tree.

Only SDK-shaped calls are considered - the ``unitree_sdk2py`` surface is
PascalCase and everything written in this repo is snake_case, so the case of
the callee separates a bus operation from a local helper without a hand-kept
list of either. ``Init`` is a generic name; a future unrelated ``.Init()``
being flagged here is intended, because on this tree that name has only ever
meant a DDS endpoint.

The graded set is the whole ``strands_robots`` package, not just
``tools/g1``: the lock's second consumer is the tools, but its first is the
driver, and nothing stops a future mesh or driver module from constructing an
endpoint directly. Today none do - they reach the bus through
``_dds_engine`` - so grading the whole tree costs nothing and catches the
next one.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import threading
from typing import Any

import pytest

import strands_robots
import strands_robots.tools.g1 as g1_pkg
from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK

# ``strands_robots.tools.g1`` lazy-exports a ``use_unitree`` *tool* under the
# same name as the module declaring it, so an attribute import would bind the
# ``@tool`` object and not the module whose privates these cells reach into.
use_unitree = importlib.import_module("strands_robots.tools.g1.use_unitree")

_LOCK_NAME = "_DDS_INIT_LOCK"
_PKG_ROOT = pathlib.Path(strands_robots.__file__).parent
_G1_TOOLS_DIR = pathlib.Path(g1_pkg.__file__).parent


def _is_the_shared_lock(node: ast.expr) -> bool:
    """True when an expression names the shared lock, imported or attributed."""
    if isinstance(node, ast.Name):
        return node.id == _LOCK_NAME
    if isinstance(node, ast.Attribute):
        return node.attr == _LOCK_NAME
    return False


def _guards_the_shared_lock(node: ast.stmt) -> bool:
    """True when a ``with``/``async with`` statement acquires the shared lock."""
    if not isinstance(node, (ast.With, ast.AsyncWith)):
        return False
    return any(_is_the_shared_lock(item.context_expr) for item in node.items)


def _callee_name(call: ast.Call) -> str | None:
    """The final identifier of a call's callee - ``a.b.C()`` -> ``C``."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_sdk_shaped(name: str) -> bool:
    """The SDK's surface is PascalCase; everything in this repo is snake_case."""
    return bool(name) and name[0].isupper()


def _sdk_calls_by_guard(source: str) -> tuple[set[str], set[str]]:
    """Partition a module's SDK-shaped calls into guarded and unguarded.

    Args:
        source: Python source text.

    Returns:
        ``(guarded, unguarded)`` - the SDK-shaped callee names appearing
        lexically inside a ``with _DDS_INIT_LOCK`` block, and those appearing
        outside every such block.
    """
    guarded: set[str] = set()
    unguarded: set[str] = set()

    def walk(node: ast.AST, held: bool) -> None:
        for child in ast.iter_child_nodes(node):
            now_held = held or _guards_the_shared_lock(child)  # type: ignore[arg-type]
            if isinstance(child, ast.Call):
                name = _callee_name(child)
                if name is not None and _is_sdk_shaped(name):
                    (guarded if now_held else unguarded).add(name)
            walk(child, now_held)

    walk(ast.parse(source), False)
    return guarded, unguarded


def _unguarded_sites(source: str, operations: set[str]) -> list[tuple[str, int]]:
    """Every call to one of ``operations`` that sits outside the shared lock."""
    sites: list[tuple[str, int]] = []

    def walk(node: ast.AST, held: bool) -> None:
        for child in ast.iter_child_nodes(node):
            now_held = held or _guards_the_shared_lock(child)  # type: ignore[arg-type]
            if isinstance(child, ast.Call) and not now_held:
                name = _callee_name(child)
                if name is not None and name in operations:
                    sites.append((name, child.lineno))
            walk(child, now_held)

    walk(ast.parse(source), False)
    return sites


def _contract_owner_sources() -> dict[str, str]:
    """The private ``tools/g1`` modules that hold the lock - the contract owners."""
    owners: dict[str, str] = {}
    for path in sorted(_G1_TOOLS_DIR.glob("_*.py")):
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8")
        if any(
            _guards_the_shared_lock(node)
            for node in ast.walk(ast.parse(source))  # type: ignore[arg-type]
        ):
            owners[path.name] = source
    return owners


def _endpoint_creating_operations() -> set[str]:
    """Operations the contract owners only ever perform under the shared lock."""
    inside: set[str] = set()
    outside: set[str] = set()
    for source in _contract_owner_sources().values():
        guarded, unguarded = _sdk_calls_by_guard(source)
        inside |= guarded
        outside |= unguarded
    return inside - outside


def _package_sources() -> dict[str, str]:
    """Every module in ``strands_robots``, keyed by path relative to the package."""
    return {
        str(path.relative_to(_PKG_ROOT)): path.read_text(encoding="utf-8") for path in sorted(_PKG_ROOT.rglob("*.py"))
    }


class TestTheRuleHasASubject:
    """The derivation must not silently yield nothing, which would pass always."""

    def test_the_contract_owners_are_found(self) -> None:
        owners = _contract_owner_sources()
        assert "_g1_common.py" in owners, (
            "the module defining _DDS_INIT_LOCK no longer holds it; the "
            "derivation below reads the contract out of these modules"
        )
        assert "_dds_engine.py" in owners

    @pytest.mark.parametrize(
        "operation",
        ["ChannelSubscriber", "ChannelPublisher", "ChannelFactoryInitialize", "Init"],
    )
    def test_the_known_endpoint_operations_are_derived(self, operation: str) -> None:
        assert operation in _endpoint_creating_operations(), (
            f"{operation} dropped out of the derived set, so the rule below no "
            "longer grades it. Either it is now performed outside the lock in "
            "an infrastructure module, or it is gone."
        )


class TestEveryEndpointConstructionIsUnderTheSharedLock:
    """The rule: no module performs an endpoint operation outside the lock."""

    def test_no_module_creates_an_endpoint_outside_the_shared_lock(self) -> None:
        operations = _endpoint_creating_operations()
        violations = {
            name: sites
            for name, source in _package_sources().items()
            if (sites := _unguarded_sites(source, operations))
        }
        assert not violations, (
            "these call sites construct a DDS endpoint without holding the "
            f"shared {_LOCK_NAME}, so they race every other endpoint "
            "construction in the process and the loss is a native segfault: "
            f"{violations}"
        )

    def test_the_rule_reports_an_unguarded_construction(self) -> None:
        """The shape this file was written for must still be caught."""
        before_the_fix = (
            "def _get_client(service_name):\n"
            "    with _CLIENTS_LOCK:\n"
            "        client = cls()\n"
            "        client.SetTimeout(timeout)\n"
            "        client.Init()\n"
            "        return client\n"
        )
        sites = _unguarded_sites(before_the_fix, _endpoint_creating_operations())
        assert ("Init", 5) in sites, (
            "the rule no longer flags an Init() taking only a private lock, "
            "which is the exact shape it exists to refuse"
        )

    def test_the_rule_accepts_a_guarded_construction(self) -> None:
        after_the_fix = (
            "def _get_client(service_name):\n"
            "    with _CLIENTS_LOCK:\n"
            "        client = cls()\n"
            "        with _DDS_INIT_LOCK:\n"
            "            client.Init()\n"
            "        return client\n"
        )
        assert not _unguarded_sites(after_the_fix, _endpoint_creating_operations())


class _FakeClient:
    """An SDK client that records whether the shared lock was held for it."""

    def __init__(self) -> None:
        self.lock_held_during: dict[str, bool] = {"construct": _DDS_INIT_LOCK.locked()}

    def SetTimeout(self, timeout: float) -> None:  # noqa: N802 - SDK spelling
        self.lock_held_during["set_timeout"] = _DDS_INIT_LOCK.locked()

    def Init(self) -> None:  # noqa: N802 - SDK spelling
        self.lock_held_during["init"] = _DDS_INIT_LOCK.locked()


@pytest.fixture
def _no_cached_clients() -> Any:
    """``_get_client`` only constructs on a cache miss."""
    use_unitree._CLIENTS.clear()
    yield
    use_unitree._CLIENTS.clear()


class TestGetClientConstructsUnderTheSharedLock:
    """The lexical rule is necessary but not sufficient - grade the behaviour."""

    def test_the_lock_is_held_for_every_construction_step(
        self, monkeypatch: pytest.MonkeyPatch, _no_cached_clients: None
    ) -> None:
        monkeypatch.setattr(use_unitree, "_import_client_class", lambda qualname: _FakeClient)
        client = use_unitree._get_client("loco")
        assert client.lock_held_during == {
            "construct": True,
            "set_timeout": True,
            "init": True,
        }

    def test_a_thread_holding_the_lock_defers_the_construction(
        self, monkeypatch: pytest.MonkeyPatch, _no_cached_clients: None
    ) -> None:
        """Mutual exclusion against the engine, not just against this path.

        The engine constructs subscribers under the shared lock on its own
        threads. If ``_get_client`` took only its private lock, this
        construction would proceed while that one was in flight - which is
        the segfault. So: hold the lock, and the construction must not begin.
        """
        monkeypatch.setattr(use_unitree, "_import_client_class", lambda qualname: _FakeClient)
        constructed = threading.Event()

        def build() -> None:
            use_unitree._get_client("loco")
            constructed.set()

        with _DDS_INIT_LOCK:
            worker = threading.Thread(target=build, daemon=True)
            worker.start()
            began_while_held = constructed.wait(timeout=0.5)

        worker.join(timeout=5.0)
        assert not began_while_held, (
            "the client was constructed while another thread held the shared "
            "lock, so this path is not serialised against the engine's "
            "endpoint construction"
        )
        assert constructed.wait(timeout=5.0), "the construction never completed after the lock was released"
