"""A missing ``unitree_sdk2py`` is refused with the install line, everywhere.

``unitree_sdk2py`` is a vendor SDK that no extra of this project declares, and
it cannot be one: the PyPI ``unitree-sdk2`` wheel lacks its ``g1`` package and
pins ``cyclonedds==0.10.2``, which has no wheel for the Python this project
requires. So the refusal is the only place a user learns how to get it. Before
this test every site said ``unitree_sdk2py is not installed: <exc>`` and
stopped - the ``booster`` driver, on the same vendor-wheel footing, already
named ``pip install booster_robotics_sdk_python``.

Two things are pinned here:

* the shared text, :func:`strands_robots.tools.g1._g1_common.sdk_missing`,
  names the install line, the platform caveat and the doc section, and keeps
  the original exception verbatim (a half-installed SDK fails differently from
  an absent one, and that difference is the diagnosis);
* every lazy ``unitree_sdk2py`` import in the tree routes its ``ImportError``
  through that one function, read off the source with :mod:`ast` - so a new
  import site that hand-rolls the bare string fails here, not on a robot.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from strands_robots.drivers.go2 import Go2Driver
from strands_robots.tools.g1 import _g1_common
from strands_robots.tools.g1._dds_engine import DDSSubscriberSet
from strands_robots.tools.g1._g1_common import UNITREE_SDK_INSTALL, ensure_dds, reset_dds_state, sdk_missing
from tests.drivers.test_go2_driver import _released_driver, _text, install_unitree_sdk_stub

_PACKAGE = Path(_g1_common.__file__).resolve().parents[2]

#: Every fragment a missing-SDK answer must carry to be actionable.
_REQUIRED = (
    "unitree_sdk2py is not installed",
    "pip install",
    "unitree_sdk2_python",
    "--no-deps",
    "cyclonedds",
    "CYCLONEDDS_HOME",
    "humanoids.md",
)


def test_the_text_names_the_install_line_and_keeps_the_exception() -> None:
    exc = ImportError("No module named 'unitree_sdk2py'")

    text = sdk_missing(exc)

    for fragment in _REQUIRED:
        assert fragment in text, fragment
    assert "No module named 'unitree_sdk2py'" in text
    assert UNITREE_SDK_INSTALL in text


def test_the_install_line_is_the_recipe_that_was_proven() -> None:
    """The three commands, in order; a wheel for the binding, a checkout for the SDK."""
    steps = [s.strip() for s in UNITREE_SDK_INSTALL.split("&&")]

    assert steps[0].startswith("pip install 'cyclonedds>=0.10.2,<12'")
    assert steps[1] == "git clone https://github.com/unitreerobotics/unitree_sdk2_python"
    assert steps[2] == "pip install --no-deps -e ./unitree_sdk2_python"


def test_ensure_dds_without_the_sdk_answers_with_the_install_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first SDK touch every G1 path makes is the one a fresh install hits."""
    reset_dds_state()
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", None)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core", None)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", None)

    reason = ensure_dds("lo")

    assert reason is not None
    assert reason.startswith("unitree_sdk2py is not installed: ")
    assert UNITREE_SDK_INSTALL in reason
    reset_dds_state()


def test_a_subscriber_set_without_the_sdk_answers_with_the_install_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", None)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core", None)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", None)
    subs = DDSSubscriberSet("lo")
    subs._started = True

    reason = subs.subscribe("rt/lowstate", object, lambda _msg: None)

    assert reason is not None
    assert UNITREE_SDK_INSTALL in reason


def test_a_go2_write_without_the_sdk_sealer_answers_with_the_install_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal a driver verb returns carries the same line as the engine's."""
    install_unitree_sdk_stub(monkeypatch)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.utils.crc", None)
    driver: Go2Driver
    driver, pub = _released_driver()

    result = driver.send_action({"FL_hip_joint": 0.25})

    assert result["status"] == "error"
    assert "unitree_sdk2py is not installed" in _text(result)
    assert UNITREE_SDK_INSTALL in _text(result)
    assert pub.writes == []


def _handlers_around(tree: ast.Module, lineno: int) -> list[ast.ExceptHandler]:
    """The ``except`` clauses of the innermost ``try`` enclosing *lineno*."""
    best: ast.Try | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.lineno <= lineno <= (node.end_lineno or node.lineno):
            body_end = max(getattr(stmt, "end_lineno", stmt.lineno) for stmt in node.body)
            if node.lineno <= lineno <= body_end and (best is None or node.lineno > best.lineno):
                best = node
    return best.handlers if best is not None else []


def _imports_unitree(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "unitree_sdk2py" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "unitree_sdk2py"
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module":
        first = node.args[0] if node.args else None
        return isinstance(first, ast.Constant) and str(first.value).startswith("unitree_sdk2py")
    return False


def _sites() -> list[tuple[Path, int, list[ast.ExceptHandler]]]:
    found: list[tuple[Path, int, list[ast.ExceptHandler]]] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if _imports_unitree(node):
                found.append((path, node.lineno, _handlers_around(tree, node.lineno)))
    return found


def test_every_lazy_import_site_routes_its_import_error_through_sdk_missing() -> None:
    """Read off the source: a handler that catches ImportError must call ``sdk_missing``.

    Sites whose ``try`` has no ImportError handler are the ones that let the
    error propagate to a caller that has one (``_motion_switcher.load_client``
    is reached only after :func:`ensure_dds` succeeded) - those are allowed;
    what is refused is a handler that *answers* the ImportError with text of
    its own.
    """
    sites = _sites()
    assert len(sites) >= 12, [f"{p.name}:{n}" for p, n, _ in sites]

    offenders: list[str] = []
    for path, lineno, handlers in sites:
        for handler in handlers:
            names = {
                getattr(t, "id", "")
                for t in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type])
                if t is not None
            }
            if not names & {"ImportError", "ModuleNotFoundError"}:
                continue
            source = ast.unparse(handler)
            if "sdk_missing(" not in source:
                offenders.append(f"{path.relative_to(_PACKAGE.parent)}:{lineno}")

    assert offenders == [], offenders
