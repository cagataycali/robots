"""A missing ``unitree_sdk2py`` is refused with the install line, everywhere.

``unitree_sdk2py`` is a vendor SDK that no extra of this project declares, and
it cannot be one: the PyPI ``unitree-sdk2`` wheel lacks its ``g1`` package and
pins ``cyclonedds==0.10.2``, which has no wheel for the Python this project
requires. So the refusal is the only place a user learns how to get it. Before
this test every site said ``unitree_sdk2py is not installed: <exc>`` and
stopped - the ``booster`` driver, on the same vendor-wheel footing, already
named ``pip install booster_robotics_sdk_python``.

Two things are pinned here:

* the shared text, :func:`strands_robots.drivers.unitree._common.sdk_missing`,
  names the install line, the platform caveat and the doc section, and keeps
  the original exception verbatim (a half-installed SDK fails differently from
  an absent one, and that difference is the diagnosis);
* every lazy ``unitree_sdk2py`` import in the tree routes its ``ImportError``
  through that one function, read off the source with :mod:`ast` - so a new
  import site that hand-rolls the bare string fails here, not on a robot.

The source scan reads three kinds of site, because the first two kinds of
blindness each hid a live refusal. A handler that catches ``Exception`` catches
the ``ImportError`` too, so it is the handler that answers and it owes the same
text. And an import reached through a helper that lets the error propagate -
``_load_motion_switcher_client`` - is answered by the *caller's* handler, in a
file with no ``unitree_sdk2py`` in it at all. The partial install is what makes
those two reachable: the PyPI ``unitree-sdk2`` wheel ships no ``comm`` package,
so the bus init and the IDL classes succeed, ``connect_eagerly`` returns
``None``, and the motion-switcher open is the only thing that fails - on both
robots, at the gate that hands over the legs.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import sys
from pathlib import Path

import pytest

from strands_robots.drivers.g1 import G1Driver
from strands_robots.drivers.g1 import _resolve_message_class as _g1_resolve
from strands_robots.drivers.go2 import Go2Driver
from strands_robots.drivers.go2 import _resolve_message_class as _go2_resolve
from strands_robots.drivers.unitree import _common
from strands_robots.drivers.unitree._common import UNITREE_SDK_INSTALL, ensure_dds, reset_dds_state, sdk_missing
from strands_robots.drivers.unitree._dds_engine import DDSSubscriberSet
from tests.drivers.test_go2_driver import _released_driver, _text, install_unitree_sdk_stub

_PACKAGE = Path(_common.__file__).resolve().parents[2]

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
    """The three commands, in order; this project's extra for the binding, a checkout for the SDK.

    The binding step names ``[ros2]`` rather than a bare ``cyclonedds`` range.
    That extra declares the same requirement, so the manifest owns the version
    and a refusal read off a robot cannot quote a bound the project has moved -
    which is the rule ``tests/test_ur_rtde_extra_is_declared.py`` grades across
    the whole drivers tree. Only the vendor SDK stays a literal command: it has
    no usable PyPI build, so no requirement can carry it.
    """
    steps = [s.strip() for s in UNITREE_SDK_INSTALL.split("&&")]

    assert steps[0] == "pip install 'strands-robots[ros2]'"
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


def _without_comm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the motion-switcher module un-importable, as the PyPI wheel does.

    The wheel ships no ``comm`` package, so this is the partial install a user
    who followed PyPI rather than the docs actually runs: every other SDK touch
    on the connect path succeeds.
    """
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.comm.motion_switcher.motion_switcher_client", None)


def test_a_g1_motion_switcher_open_without_comm_answers_with_the_install_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_status`` is where this refusal surfaces, and it is the only one there.

    The open is wrapped in ``except Exception`` - the SDK's own failures are
    opaque - and that handler catches the ImportError as well, so it is the
    handler that answers a missing SDK.
    """
    _without_comm(monkeypatch)
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True

    driver._refresh_fsm_id()
    reported = asyncio.run(driver.get_status())["content"][0]["json"]["motion_switcher_open_error"]

    assert reported is not None
    assert UNITREE_SDK_INSTALL in reported
    assert "unitree_sdk2py.comm" in reported


def test_the_go2_sport_mode_gate_without_comm_answers_with_the_install_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``release_sport_mode`` is the verb that hands the legs over, so it owes the remedy.

    Its import is indirected through ``_load_motion_switcher_client``, which
    lets the ImportError propagate - the answering handler lives in the Go2
    driver, which imports nothing from the SDK at that line.
    """
    _without_comm(monkeypatch)
    driver = Go2Driver(tool_name="go2", port="192.168.123.161")
    driver._connected = True

    result = driver.release_sport_mode(attempts=1)

    assert result["status"] == "error"
    assert UNITREE_SDK_INSTALL in _text(result)


@pytest.mark.parametrize(
    ("resolve", "module_path"),
    [
        (_g1_resolve, "unitree_sdk2py.idl.unitree_hg.msg.dds_"),
        (_go2_resolve, "unitree_sdk2py.idl.unitree_go.msg.dds_"),
    ],
    ids=["g1", "go2"],
)
def test_resolving_an_idl_class_without_the_sdk_names_the_module_and_the_install_line(
    monkeypatch: pytest.MonkeyPatch,
    resolve: object,
    module_path: str,
) -> None:
    """The connect step that turns a subscription plan into IDL classes.

    The exception here is raised for a *different* module than the one asked
    for, which is what a partial install does: the deepest missing module is
    the one named, so the module being resolved has to be carried by this site
    or it is lost.
    """
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'unitree_sdk2py.idl'")),
    )

    reason = resolve((module_path, "LowState_"))  # type: ignore[operator]

    assert isinstance(reason, str)
    assert UNITREE_SDK_INSTALL in reason
    assert module_path in reason
    assert "unitree_sdk2py.idl'" in reason


#: Exception names whose handler catches an ``ImportError`` - so the first
#: handler naming one of these is the handler that answers a missing SDK.
_ANSWERING = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    caught = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return {getattr(node, "id", "") for node in caught if node is not None}


def _answering_handler(tree: ast.Module, lineno: int) -> ast.ExceptHandler | None:
    """The handler that would answer an ``ImportError`` raised at *lineno*.

    The innermost ``try`` whose *body* contains the line, then its first handler
    that catches ImportError - by that name or by catching ``Exception``. Python
    runs the first matching handler, so that one is the answer and no later one
    can add to it.
    """
    best: ast.Try | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            body_end = max(getattr(stmt, "end_lineno", stmt.lineno) for stmt in node.body)
            if node.lineno <= lineno <= body_end and (best is None or node.lineno > best.lineno):
                best = node
    if best is None:
        return None
    for handler in best.handlers:
        if _handler_names(handler) & _ANSWERING:
            return handler
    return None


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "some.module"`` bindings, for ``import_module(NAME)``."""
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            found.update({t.id: node.value.value for t in node.targets if isinstance(t, ast.Name)})
    return found


def _imports_unitree(node: ast.AST, constants: dict[str, str]) -> bool:
    """Whether *node* imports ``unitree_sdk2py``, spelled any of three ways."""
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "unitree_sdk2py" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "unitree_sdk2py"
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module":
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant):
            return str(first.value).startswith("unitree_sdk2py")
        # ``import_module(_SDK_MODULE)`` - resolve the name to its constant.
        if isinstance(first, ast.Name):
            return constants.get(first.id, "").startswith("unitree_sdk2py")
    return False


def _lineno(node: ast.AST) -> int:
    """The line a node sits on. Imports and calls - the only kinds scanned - carry one."""
    return int(getattr(node, "lineno", 0))


def _called_name(node: ast.Call) -> str:
    return str(getattr(node.func, "id", getattr(node.func, "attr", "")))


def _trees() -> dict[Path, ast.Module]:
    return {path: ast.parse(path.read_text(encoding="utf-8")) for path in sorted(_PACKAGE.rglob("*.py"))}


def _propagating_loaders(trees: dict[Path, ast.Module]) -> set[str]:
    """Functions that import the SDK and let the ``ImportError`` out.

    Their callers are the sites that answer it, which is how a refusal ends up
    in a file containing no ``unitree_sdk2py`` at all.
    """
    loaders: set[str] = set()
    for tree in trees.values():
        constants = _module_constants(tree)
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if _imports_unitree(node, constants) and _answering_handler(tree, _lineno(node)) is None:
                    loaders.add(func.name)
    return loaders


def _sites() -> list[tuple[Path, int, ast.ExceptHandler | None]]:
    """Every place a missing-SDK ``ImportError`` can arise, direct or indirect."""
    trees = _trees()
    loaders = _propagating_loaders(trees)
    found: list[tuple[Path, int, ast.ExceptHandler | None]] = []
    for path, tree in trees.items():
        constants = _module_constants(tree)
        for node in ast.walk(tree):
            indirect = isinstance(node, ast.Call) and _called_name(node) in loaders
            if _imports_unitree(node, constants) or indirect:
                found.append((path, _lineno(node), _answering_handler(tree, _lineno(node))))
    return found


def test_a_site_whose_own_line_never_names_the_sdk_is_still_in_the_roster() -> None:
    """A propagating loader's caller is a site, though its line imports nothing.

    Without this the scan reads the source for ``unitree_sdk2py``, finds none at
    the Go2's sport-mode gate, and concludes it is not an SDK site - which is
    how that gate's bare refusal survived a guard written to prevent exactly it.
    """
    assert "_load_motion_switcher_client" in _propagating_loaders(_trees())
    go2 = _PACKAGE / "drivers" / "go2.py"
    lines = go2.read_text(encoding="utf-8").splitlines()

    indirect = [f"go2.py:{n}" for path, n, _h in _sites() if path == go2 and "unitree_sdk2py" not in lines[n - 1]]

    assert indirect, "no indirect site found in go2.py; the scan is only reading imports"


@pytest.mark.parametrize(
    ("handler", "answers"),
    [
        ("except ImportError as exc:\n    pass\n", True),
        ("except ModuleNotFoundError as exc:\n    pass\n", True),
        # ``except Exception`` catches the ImportError too, so it is the answer.
        ("except Exception as exc:\n    pass\n", True),
        ("except BaseException as exc:\n    pass\n", True),
        ("except (ValueError, ImportError) as exc:\n    pass\n", True),
        ("except ValueError as exc:\n    pass\n", False),
        ("finally:\n    pass\n", False),
    ],
    ids=["import-error", "module-not-found", "broad", "base", "tuple", "unrelated", "no-handler"],
)
def test_the_answering_handler_is_the_first_one_that_would_catch_an_import_error(handler: str, answers: bool) -> None:
    """Python runs the first matching handler, so that one is the whole answer.

    A handler catching ``Exception`` answers a missing SDK as surely as one
    naming ``ImportError``; reading only the latter is what left two refusals
    bare.
    """
    source = "try:\n    from unitree_sdk2py.core.channel import ChannelPublisher\n" + handler

    found = _answering_handler(ast.parse(source), 2)

    assert (found is not None) is answers


def test_the_answering_handler_is_the_first_match_not_the_broadest() -> None:
    """An earlier unrelated handler does not shadow the one that catches it."""
    source = (
        "try:\n    from unitree_sdk2py.core.channel import ChannelPublisher\n"
        "except ValueError:\n    pass\n"
        "except Exception as exc:\n    answer = sdk_missing(exc)\n"
    )

    handler = _answering_handler(ast.parse(source), 2)

    assert handler is not None
    assert "sdk_missing(" in ast.unparse(handler)


def test_every_lazy_import_site_routes_its_import_error_through_sdk_missing() -> None:
    """Read off the source: whichever handler answers the ImportError must call ``sdk_missing``.

    Sites with no answering handler are the ones that let the error reach a
    caller that has one (``_load_motion_switcher_client``, reached only after
    :func:`ensure_dds` succeeded) - those are allowed; what is refused is a
    handler that *answers* the ImportError with text of its own.
    """
    sites = _sites()
    assert len(sites) >= 12, [f"{p.name}:{n}" for p, n, _ in sites]

    offenders = [
        f"{path.relative_to(_PACKAGE.parent)}:{lineno}"
        for path, lineno, handler in sites
        if handler is not None and "sdk_missing(" not in ast.unparse(handler)
    ]

    assert offenders == [], offenders
