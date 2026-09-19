"""A stand-in ``device_connect_edge`` installed at collection time is not handed back.

Two test modules install ``MagicMock`` stand-ins for ``device_connect_edge`` at
import time and re-import ``strands_robots.device_connect.*`` under them, undoing
both at ``teardown_module``. Collection of every file precedes the teardown of
any, so what each one's snapshot records, and what every later-collected file
binds, is decided by the order files collect in - and two orderings that the
serial alphabetical run happens to avoid failed the same four cells.

Selected ahead of ``tests/drivers/test_reachy_wireless_daemon_protocol.py``, the
first installing file's snapshot predates that file's import of
``reachy_transport`` - a module that imports nothing from the edge - and the
teardown that put the snapshot back dropped it. The reachy file then patched an
orphan, the driver imported a fresh copy, and a unit test reported ``daemon
unreachable (reachy-a.local:8000)`` after resolving the hostname for real. And
selected together, the second installing file collected while the first one's
stand-ins were resident, so the "originals" it recorded were the first file's
``MagicMock`` and an integration bound to the first file's ``DeviceDriver``
stand-in - which is what its teardown handed back, to every import after it.

Both are the one owner, :mod:`tests._device_connect_real`, reading a snapshot
by *what it holds* rather than by when it was taken: a real module goes back, a
module bound to a fake does not, and a newer real module is left where the
sibling that imported it can still reach it. The cells here grade each rule on
a synthetic package, then the two measured orderings on the real files.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from tests._device_connect_real import (
    bound_to_a_fake,
    held_modules,
    restore,
    restore_the_edge,
    use_a_mock_edge,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: A package shaped like the integration: a driver bound to the edge's base
#: class, a leaf that imports nothing from the edge, and a facade whose function
#: closes over the driver module - the three shapes a snapshot can hold.
_PACKAGE = {
    "probe_edge/__init__.py": "class DeviceDriver:\n    pass\n",
    "probe_dc/__init__.py": "",
    "probe_dc/driver.py": "from probe_edge import DeviceDriver\n\n\nclass Driver(DeviceDriver):\n    pass\n",
    "probe_dc/leaf.py": 'def api():\n    return "real"\n',
    "probe_dc/facade.py": "from probe_dc.driver import Driver\n\n\ndef init():\n    return Driver()\n",
}

_PREFIX = "probe_dc"
_PACKAGE_MODULES = (_PREFIX, f"{_PREFIX}.driver", f"{_PREFIX}.leaf", f"{_PREFIX}.facade")


class _StandInDriver:
    """The base class a mock file supplies in place of ``DeviceDriver``."""


@pytest.fixture
def probe_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The synthetic edge and integration on ``sys.path``, both taken away after."""
    for relative, source in _PACKAGE.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("probe_edge", *_PACKAGE_MODULES):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield tmp_path
    for name in ("probe_edge", *_PACKAGE_MODULES):
        sys.modules.pop(name, None)


def _import_bound_to_a_stand_in(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``probe_dc.<name>`` fresh while a stand-in edge is registered."""
    stand_in = ModuleType("probe_edge")
    stand_in.DeviceDriver = _StandInDriver  # type: ignore[attr-defined]
    with monkeypatch.context() as edge:
        edge.setitem(sys.modules, "probe_edge", stand_in)
        for module_name in _PACKAGE_MODULES[1:]:
            edge.delitem(sys.modules, module_name, raising=False)
        module = importlib.import_module(f"{_PREFIX}.{name}")
    return module


class TestWhatASnapshotHolds:
    """``bound_to_a_fake`` reads the binding, not the time it was made."""

    def test_a_module_bound_to_the_real_edge_is_real(self, probe_package: Path) -> None:
        assert bound_to_a_fake(importlib.import_module(f"{_PREFIX}.driver"), _PREFIX) is False

    def test_a_module_importing_nothing_from_the_edge_is_real(self, probe_package: Path) -> None:
        assert bound_to_a_fake(importlib.import_module(f"{_PREFIX}.leaf"), _PREFIX) is False

    def test_a_driver_subclassing_a_stand_in_is_bound_to_a_fake(
        self, probe_package: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver = _import_bound_to_a_stand_in("driver", monkeypatch)
        assert driver.Driver.__mro__[1] is _StandInDriver, "the premise: the import bound the stand-in"
        assert bound_to_a_fake(driver, _PREFIX) is True

    def test_a_mock_read_off_a_stand_in_is_bound_to_a_fake(self, probe_package: Path) -> None:
        leaf = importlib.import_module(f"{_PREFIX}.leaf")
        leaf.DeviceRuntime = MagicMock()  # type: ignore[attr-defined]
        assert bound_to_a_fake(leaf, _PREFIX) is True

    def test_a_function_from_a_bound_sibling_carries_the_binding(
        self, probe_package: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The package caches ``init_device_connect`` from ``_impl``: a function, not a class."""
        facade = _import_bound_to_a_stand_in("facade", monkeypatch)
        package = ModuleType(_PREFIX)
        package.init = facade.init  # type: ignore[attr-defined]
        assert bound_to_a_fake(package, _PREFIX) is True

    def test_a_function_from_outside_the_prefix_is_not_followed(
        self, probe_package: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        facade = _import_bound_to_a_stand_in("facade", monkeypatch)
        package = ModuleType(_PREFIX)
        package.init = facade.init  # type: ignore[attr-defined]
        assert bound_to_a_fake(package, "some_other_package") is False


class TestRestoreReadsTheSnapshot:
    """Each name under the prefix gets one of three answers."""

    def test_a_newer_real_module_is_left_where_the_sibling_can_reach_it(self, probe_package: Path) -> None:
        """The reachy ordering: ``reachy_transport`` imported after the snapshot was dropped."""
        held = held_modules(_PREFIX)
        assert held == {}
        leaf = importlib.import_module(f"{_PREFIX}.leaf")
        package = sys.modules[_PREFIX]

        restore(held, _PREFIX)

        assert sys.modules[f"{_PREFIX}.leaf"] is leaf
        assert sys.modules[_PREFIX] is package
        assert package.leaf is leaf  # type: ignore[attr-defined]

    def test_a_real_snapshot_goes_back_over_a_newer_copy(self, probe_package: Path) -> None:
        """The contract from before: the object every earlier-collected module holds wins."""
        first = importlib.import_module(f"{_PREFIX}.leaf")
        held = held_modules(_PREFIX)
        for name in _PACKAGE_MODULES:
            sys.modules.pop(name, None)
        second = importlib.import_module(f"{_PREFIX}.leaf")
        assert second is not first

        restore(held, _PREFIX)

        assert sys.modules[f"{_PREFIX}.leaf"] is first
        assert sys.modules[_PREFIX].leaf is first  # type: ignore[attr-defined]

    def test_a_snapshot_bound_to_a_fake_is_not_handed_back(
        self, probe_package: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second installing file's snapshot: taken while the first one's stand-in was resident."""
        bound = _import_bound_to_a_stand_in("driver", monkeypatch)
        held = {_PREFIX: sys.modules[_PREFIX], f"{_PREFIX}.driver": bound}
        for name in _PACKAGE_MODULES:
            sys.modules.pop(name, None)
        real = importlib.import_module(f"{_PREFIX}.driver")

        restore(held, _PREFIX)

        assert sys.modules[f"{_PREFIX}.driver"] is real
        assert sys.modules[_PREFIX].driver is real  # type: ignore[attr-defined]

    def test_a_newer_module_bound_to_a_fake_is_dropped_with_its_parent_attribute(
        self, probe_package: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``from pkg import leaf`` serves the attribute when it exists, so both bindings go."""
        importlib.import_module(_PREFIX)
        held = held_modules(_PREFIX)
        bound = _import_bound_to_a_stand_in("driver", monkeypatch)
        package = sys.modules[_PREFIX]
        assert package.driver is bound  # type: ignore[attr-defined]

        restore(held, _PREFIX)

        assert f"{_PREFIX}.driver" not in sys.modules
        assert not hasattr(package, "driver")
        fresh = importlib.import_module(f"{_PREFIX}.driver")
        assert fresh is not bound
        assert bound_to_a_fake(fresh, _PREFIX) is False


_EDGE_NAMES = (
    "device_connect_edge",
    "device_connect_edge.drivers",
    "device_connect_edge.types",
    "device_connect_edge.device",
)


def _stand_ins() -> dict[str, MagicMock]:
    return {name: MagicMock(name=name) for name in _EDGE_NAMES}


class TestTheEdgeSnapshotRecordsOnlyWhatIsReal:
    """``use_a_mock_edge`` / ``restore_the_edge``: the edge names themselves."""

    @pytest.fixture(autouse=True)
    def _edge_names_put_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _EDGE_NAMES:
            if name in sys.modules:
                monkeypatch.setitem(sys.modules, name, sys.modules[name])
            else:
                monkeypatch.delitem(sys.modules, name, raising=False)

    def test_a_siblings_stand_in_is_not_recorded_as_the_original(self) -> None:
        first = _stand_ins()
        for name, stand_in in first.items():
            sys.modules[name] = stand_in

        swap = use_a_mock_edge(_stand_ins())

        assert swap.edge == dict.fromkeys(_EDGE_NAMES)

    def test_a_real_module_is_recorded_and_handed_back(self) -> None:
        real = {name: ModuleType(name) for name in _EDGE_NAMES}
        for name, module in real.items():
            module.__file__ = f"/real/{name}.py"
            sys.modules[name] = module

        swap = use_a_mock_edge(_stand_ins())
        assert swap.edge == real

        restore_the_edge(swap)
        assert {name: sys.modules[name] for name in _EDGE_NAMES} == real

    def test_a_real_module_registered_meanwhile_is_never_displaced(self) -> None:
        swap = use_a_mock_edge(_stand_ins())
        real = ModuleType("device_connect_edge")
        real.__file__ = "/real/device_connect_edge.py"
        sys.modules["device_connect_edge"] = real

        restore_the_edge(swap)

        assert sys.modules["device_connect_edge"] is real
        assert {name: sys.modules.get(name) for name in _EDGE_NAMES[1:]} == {
            name: swap.edge[name] for name in _EDGE_NAMES[1:]
        }


#: A test module run last, grading what the two installing files leave behind.
_WHAT_IS_LEFT = """
import sys


def test_what_the_installing_files_left_behind():
    stand_ins = {
        name: module
        for name in ("device_connect_edge", "device_connect_edge.drivers", "device_connect_edge.types", "device_connect_edge.device")
        if (module := sys.modules.get(name)) is not None and not hasattr(module, "__file__")
    }
    assert stand_ins == {}, f"a stand-in is still registered: {stand_ins}"
    bound = {
        name: module.DeviceDriver.__module__
        for name, module in sys.modules.items()
        if name.startswith("strands_robots.device_connect") and hasattr(module, "DeviceDriver")
        if not module.DeviceDriver.__module__.startswith("device_connect_edge")
    }
    assert bound == {}, f"an integration module is still bound to a stand-in: {bound}"
"""


def _run_pytest(*args: str, cwd: Path) -> str:
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(_REPO_ROOT / "pyproject.toml"),
            *args,
            "-q",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
        ],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=600,
    )
    return finished.stdout + finished.stderr


def _counted(report: str, outcome: str) -> int:
    found = re.search(rf"(\d+) {outcome}", report)
    return int(found.group(1)) if found else 0


#: Whether the extra is installed, read from ``sys.path`` rather than
#: ``sys.modules``: this file collects after the installing files, while their
#: stand-ins are registered, and ``importlib.util.find_spec`` refuses a
#: registered entry with no ``__spec__``.
_EDGE_IS_INSTALLED = importlib.machinery.PathFinder.find_spec("device_connect_edge") is not None


@pytest.mark.skipif(not _EDGE_IS_INSTALLED, reason="[device-connect] extra not installed")
class TestTheMeasuredOrderings:
    """The two selections that failed, run as they were measured."""

    def test_the_installing_file_ahead_of_the_reachy_driver_file(self) -> None:
        report = _run_pytest(
            "tests/test_device_connect_all_robots.py",
            "tests/drivers/test_reachy_wireless_daemon_protocol.py",
            "-k",
            "not websocket_close",
            cwd=_REPO_ROOT,
        )
        assert _counted(report, "failed") == 0 and _counted(report, "passed") > 0, report[-4000:]

    def test_both_installing_files_then_what_they_left_behind(self, tmp_path: Path) -> None:
        probe = tmp_path / "test_what_is_left.py"
        probe.write_text(textwrap.dedent(_WHAT_IS_LEFT), encoding="utf-8")
        report = _run_pytest(
            "tests/test_device_connect_all_robots.py",
            "tests/test_device_connect_drivers.py",
            str(probe),
            cwd=_REPO_ROOT,
        )
        assert _counted(report, "failed") == 0 and _counted(report, "passed") > 0, report[-4000:]
