"""No two callers of one description cache are inside the clone at once.

Importing ``robot_descriptions.<name>_mj_description`` clones an upstream
repository into a shared cache directory, and 40-odd descriptions name one
``mujoco_menagerie``. Upstream's ``clone_to_directory`` tests that directory for
a usable clone and then creates it, holding no lock, so two callers inside that
window both clone into one directory and the loser's ``git`` fails in a tree the
winner is building - reported as the robot's own download failure.

CPython's import lock does not close the window: it is per module, and the
colliding callers import *different* descriptions that share one repository.
So the four places the package triggers a clone all route through
:func:`strands_robots._description_cache.import_description`, and the cells
below grade that: each entry point is driven from two threads importing two
descriptions whose imports report whether they overlapped, the scan pins that no
fifth call site can skip the lock, and the two degraded conditions still import.

A test module is the other half of the window: it reaches a description through
its own ``pytest.importorskip("robot_descriptions.<name>")``, which no package
seam sees, and a distributed run collects every file in every worker. So
:mod:`tests.description_clone_lock` wraps ``clone_to_cache`` itself in *this*
module's lock for the session, and the last two cells grade that the two halves
wait on one lock file rather than on one each.
"""

from __future__ import annotations

import ast
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from strands_robots import _description_cache as dc
from strands_robots.assets import download as dl
from strands_robots.registry import discovery
from tests.description_clone_lock import INSTALLED, serialize_description_clones

# A description module whose *import* reports itself to the probe, the way a
# real one clones while it is being imported.
_DESCRIPTION_BODY = """
import _clone_probe

_clone_probe.enter()
try:
    MJCF_PATH = _clone_probe.MODEL
    URDF_PATH = _clone_probe.MODEL
    PACKAGE_PATH = _clone_probe.PACKAGE
finally:
    _clone_probe.leave()
"""

_MJCF = '<mujoco model="probe"><worldbody><geom type="box" size="1 1 1"/></worldbody></mujoco>'

#: Two descriptions, as two robots sharing one upstream repository would be.
_NAMES = ("probealpha_mj_description", "probebeta_mj_description")


class _Probe:
    """Records how many description imports were inside the clone at once."""

    def __init__(self, package: Path) -> None:
        self.PACKAGE = str(package)
        self.MODEL = str(package / "bot.xml")
        self._guard = threading.Lock()
        self._inside = 0
        self.peak = 0

    def enter(self) -> None:
        with self._guard:
            self._inside += 1
            self.peak = max(self.peak, self._inside)
        # Wide enough that an unguarded second caller lands inside the window.
        threading.Event().wait(0.15)

    def leave(self) -> None:
        with self._guard:
            self._inside -= 1


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Probe]:
    """Install two importable descriptions that report their own overlap.

    The real ``robot_descriptions`` package is replaced by one whose ``__path__``
    holds the two modules, so ``importlib.import_module`` executes them for real
    - a ``sys.modules`` stand-in would return without ever entering the window.
    """
    package = tmp_path / "package"
    package.mkdir()
    (package / "bot.xml").write_text(_MJCF)
    for name in _NAMES:
        (tmp_path / f"{name}.py").write_text(_DESCRIPTION_BODY)

    state = _Probe(package)
    module = ModuleType("_clone_probe")
    module.enter = state.enter  # type: ignore[attr-defined]
    module.leave = state.leave  # type: ignore[attr-defined]
    module.MODEL = state.MODEL  # type: ignore[attr-defined]
    module.PACKAGE = state.PACKAGE  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_clone_probe", module)

    parent = ModuleType("robot_descriptions")
    parent.__path__ = [str(tmp_path)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "robot_descriptions", parent)
    for name in _NAMES:
        monkeypatch.delitem(sys.modules, f"robot_descriptions.{name}", raising=False)
    monkeypatch.setenv(dc.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(discovery, "_DISCOVER_CACHE", {}, raising=False)
    yield state


def _fetch_the_asset(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """``download_assets``' own route: link the description's package directory."""
    info = {
        "asset": {
            "dir": name,
            "model_xml": "bot.xml",
            "scene_xml": "bot.xml",
            "robot_descriptions_module": name,
        }
    }
    return dl._download_via_robot_descriptions({name: info}, tmp_path / "assets")[name]


def _resolve_the_module(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """The naming heuristic for a robot the registry does not name a module for."""
    return dl._resolve_robot_descriptions_module(name.removesuffix("_mj_description"), {"asset": {"dir": name}})


def _discover_the_entry(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """Registry auto-discovery, which imports the module for its MJCF paths."""
    monkeypatch.setattr(discovery, "descriptions_module", lambda _n: name)
    return discovery.discover_robot(name.removesuffix("_mj_description"))


def _resolve_the_urdf(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """URDF resolution, which imports the module for its ``URDF_PATH``."""
    monkeypatch.setattr(discovery, "urdf_descriptions_module", lambda _n: name)
    return discovery.discover_urdf_path(name.removesuffix("_mj_description"))


#: Every production entry point that triggers a clone, with what a success reads.
_ENTRY_POINTS: tuple[tuple[str, Callable[..., object], object], ...] = (
    ("assets.download: fetch the asset", _fetch_the_asset, "downloaded"),
    ("assets.download: resolve the module", _resolve_the_module, None),
    ("registry.discovery: discover the entry", _discover_the_entry, None),
    ("registry.discovery: resolve the URDF", _resolve_the_urdf, None),
)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [(entry, expected) for _, entry, expected in _ENTRY_POINTS],
    ids=[name for name, _, _ in _ENTRY_POINTS],
)
def test_two_descriptions_of_one_cache_are_not_cloned_at_once(
    entry: Callable[..., object],
    expected: object,
    probe: _Probe,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both callers get their description, and neither sees the other's clone."""
    results: dict[str, object] = {}
    failures: dict[str, BaseException] = {}

    def drive(name: str) -> None:
        try:
            results[name] = entry(name, monkeypatch, tmp_path)
        except Exception as exc:  # a losing clone raises; record, do not hide
            failures[name] = exc

    threads = [threading.Thread(target=drive, args=(name,)) for name in _NAMES]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not failures, f"a concurrent caller failed: {failures}"
    assert probe.peak == 1, f"{probe.peak} callers were inside the clone at once"
    assert sorted(results) == sorted(_NAMES), f"only {sorted(results)} got a result"
    for name, result in results.items():
        assert result is not None, f"{name} resolved to nothing"
        if expected is not None:
            assert result == expected, f"{name} reported {result!r}, not {expected!r}"


def test_no_description_import_in_the_package_skips_the_lock() -> None:
    """A scan, so a fifth call site cannot be added outside the one seam.

    The raw ``importlib.import_module("robot_descriptions...")`` belongs to
    :func:`~strands_robots._description_cache.import_description` alone; every
    other module reaches a description through it.
    """
    package = Path(dc.__file__).parent
    raw: list[str] = []
    readers: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "import_description":
                readers.add(path.name)
            if not isinstance(node, ast.Call) or not node.args:
                continue
            target = ast.unparse(node.func)
            if not target.endswith("import_module"):
                continue
            if "robot_descriptions" in ast.unparse(node.args[0]):
                raw.append(f"{path.relative_to(package)}:{node.lineno}")

    assert readers >= {"download.py", "discovery.py"}, f"the seam is unread: {sorted(readers)}"
    assert raw == [f"{Path(dc.__file__).name}:{_seam_line()}"], f"a description is imported unguarded at {raw}"


def _seam_line() -> int:
    """Line of the one sanctioned raw description import."""
    source = Path(dc.__file__).read_text(encoding="utf-8").splitlines()
    return next(i for i, line in enumerate(source, 1) if "import_module(f" in line)


@pytest.mark.parametrize("degraded", ["no fcntl (Windows)", "an unusable cache directory"])
def test_a_cache_that_cannot_be_locked_still_imports(
    degraded: str, probe: _Probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone is worth more than the guard on it, so the import still happens."""
    if degraded.startswith("no fcntl"):
        monkeypatch.setattr(dc, "_HAS_FCNTL", False)
    else:
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        monkeypatch.setenv(dc.CACHE_ENV, str(blocked / "cache"))

    module: Any = dc.import_description(_NAMES[0])

    assert module.PACKAGE_PATH == probe.PACKAGE, f"{degraded}: the description did not import"


def test_the_session_installs_the_clone_lock_once() -> None:
    """conftest installed it, and installing again does not stack a second lock."""
    cache = pytest.importorskip("robot_descriptions._cache")

    assert getattr(cache.clone_to_cache, INSTALLED, False), (
        "tests/conftest.py should have installed the clone lock for the session"
    )
    installed = cache.clone_to_cache
    assert serialize_description_clones() is True
    assert cache.clone_to_cache is installed, "a second install layered another lock over the same clone"


def test_a_collecting_worker_waits_on_the_lock_a_package_caller_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One lock file for both halves: a lock each would clone into one directory together."""
    cache = pytest.importorskip("robot_descriptions._cache")
    monkeypatch.setenv(dc.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "clone_to_cache", lambda name, commit=None: f"/cache/{name}")
    assert serialize_description_clones() is True
    collecting = cache.clone_to_cache

    cloned = threading.Event()

    def collect() -> None:
        collecting("mujoco_menagerie")
        cloned.set()

    worker = threading.Thread(target=collect)
    with dc.clone_lock() as held:
        assert held == dc.cache_dir() / dc.LOCK_NAME, f"the package locks {held}"
        worker.start()
        assert not cloned.wait(0.25), "a collecting worker cloned while a package caller held the lock"
    worker.join(timeout=30)

    assert cloned.is_set(), "the collecting worker never cloned once the lock was released"
