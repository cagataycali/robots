"""The viser dashboard port is the shared TCP port domain.

``NewtonSimEngine.open_viewer`` binds the ``"viser"`` browser dashboard on a
caller-supplied ``port`` and advertises it in the returned text. Before this
domain landed the value was forwarded verbatim into ``ViewerViser(port=...)``
and interpolated into the reported URL, so every row below was accepted under
``status="success"`` and reported as a dashboard address, measured against a
recording stand-in for the viewer:

    port=0          -> success, http://localhost:0
    port=-1         -> success, http://localhost:-1
    port=65536      -> success, http://localhost:65536
    port=8080.5     -> success, http://localhost:8080.5
    port=nan        -> success, http://localhost:nan
    port=True       -> success, http://localhost:True
    port='8080'     -> success, http://localhost:8080
    port=None       -> success, http://localhost:None
    port=[8080]     -> success, http://localhost:[8080]

Two consequences, and they are why this is a defect rather than hardening.
A value that does not raise inside ``ViewerViser`` **fills the viewer slot** -
the engine holds one - so the obvious recovery, calling ``open_viewer`` again
with a usable port, is refused as "Viewer already open"; that is the same
unreusable-name shape ``test_a_refused_pose_registers_no_object`` removed for
``add_object``. And a value that *does* raise is reported as
``"Viewer launch failed: <exc>"``, which implicates the viewer rather than the
port the caller got wrong.

These tests run solver-free, against a stand-in ``self``, because the guard
precedes the lock and every viewer construction. That is deliberate: the
neighbouring ``test_viewer.py`` is gated on ``newton``/``warp`` being importable
and therefore skips wholesale on a runner without them, so a port pin placed
there would not run in CI at all.
"""

from __future__ import annotations

import ast
import inspect
import threading
import types
from pathlib import Path
from typing import Any

import pytest

import strands_robots.simulation as simulation_pkg
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import tcp_port_error

NAN = float("nan")
INF = float("inf")

#: Every value that cannot name the dashboard's port. Each row was measured as a
#: pre-fix acceptance reported as a browser URL (see the module docstring).
UNUSABLE_PORTS: tuple[Any, ...] = (
    0,
    -1,
    65536,
    99999,
    10**9,
    8080.5,
    NAN,
    INF,
    True,
    False,
    "8080",
    None,
    [8080],
)

#: Accepted ports: the documented default and both ends of the range.
USABLE_PORTS: tuple[int, ...] = (1, 8080, 65535)


class _RecordingViewer:
    """Stands in for a newton viewer and records the kwargs it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.url: str | None = None

    def set_model(self, model: Any, max_worlds: int | None = None) -> None:
        pass


def _viewer_stub(*, has_display: bool = False) -> Any:
    """A stand-in ``self`` for ``open_viewer``, with every viewer recorded.

    ``built`` collects one entry per constructed viewer, so a test can assert a
    refusal constructed nothing rather than only that it reported an error.
    """
    built: list[tuple[str, dict[str, Any]]] = []

    def _factory(kind: str) -> Any:
        def _make(**kwargs: Any) -> _RecordingViewer:
            built.append((kind, kwargs))
            return _RecordingViewer(**kwargs)

        return _make

    stub = types.SimpleNamespace(
        _world=object(),
        _model=object(),
        _viewer=None,
        _viewer_kind=None,
        _lock=threading.RLock(),
        _VIEWER_KINDS=NewtonSimEngine._VIEWER_KINDS,
        _nt=types.SimpleNamespace(
            viewer=types.SimpleNamespace(
                ViewerGL=_factory("gl"),
                ViewerViser=_factory("viser"),
                ViewerNull=_factory("null"),
            )
        ),
        _display_available=lambda: has_display,
        _sync_viewer=lambda: None,
        built=built,
    )
    return stub


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


class TestUnusablePortIsRefused:
    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_the_viser_dashboard_refuses_it(self, port: Any) -> None:
        stub = _viewer_stub()
        result = NewtonSimEngine.open_viewer(stub, "viser", port=port)
        assert result["status"] == "error", (port, result)
        assert "port" in _text(result)

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_a_refused_port_constructs_no_viewer(self, port: Any) -> None:
        """Nothing is bound and nothing is advertised for a port that cannot exist."""
        stub = _viewer_stub()
        NewtonSimEngine.open_viewer(stub, "viser", port=port)
        assert stub.built == []
        assert stub._viewer is None
        assert stub._viewer_kind is None

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_the_viewer_slot_stays_reusable(self, port: Any) -> None:
        """The obvious retry must work.

        Pre-fix a value that did not raise inside ``ViewerViser`` filled the
        single viewer slot, so the retry with a usable port came back as
        "Viewer already open" rather than opening one.
        """
        stub = _viewer_stub()
        assert NewtonSimEngine.open_viewer(stub, "viser", port=port)["status"] == "error"
        retry = NewtonSimEngine.open_viewer(stub, "viser", port=8080)
        assert retry["status"] == "success", retry
        assert "8080" in _text(retry)

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_auto_refuses_it_when_it_resolves_to_viser(self, port: Any) -> None:
        """``"auto"`` is the documented headless default, so it reaches the bind."""
        stub = _viewer_stub(has_display=False)
        assert NewtonSimEngine.open_viewer(stub, "auto", port=port)["status"] == "error"
        assert stub.built == []

    @pytest.mark.parametrize("port", (*UNUSABLE_PORTS, *USABLE_PORTS))
    def test_the_accepted_domain_is_the_shared_tcp_port_domain(self, port: Any) -> None:
        """Refuses iff the shared helper does, so the two cannot drift apart.

        The equivalence is the point: a second copy of "1-65535" here would let
        this surface and the policy providers disagree about the same port.
        """
        stub = _viewer_stub()
        refused = NewtonSimEngine.open_viewer(stub, "viser", port=port)["status"] == "error"
        assert refused is (tcp_port_error(port, "port", "NewtonSimEngine") is not None), port


class TestUsablePortIsAccepted:
    @pytest.mark.parametrize("port", USABLE_PORTS)
    def test_it_is_bound_and_advertised(self, port: int) -> None:
        stub = _viewer_stub()
        result = NewtonSimEngine.open_viewer(stub, "viser", port=port)
        assert result["status"] == "success", result
        assert stub.built == [("viser", {"port": port})]
        assert f"http://localhost:{port}" in _text(result)

    def test_the_documented_default_is_inside_the_domain(self) -> None:
        """The default must not be a value the guard would refuse."""
        default = inspect.signature(NewtonSimEngine.open_viewer).parameters["port"].default
        assert tcp_port_error(default, "port", "NewtonSimEngine") is None, default


class TestOnlyTheBoundPortIsValidated:
    """The guard runs on the branch that reads the port, and only there.

    Mirrors ``TestOnlyTheDialedPortIsValidated`` for the policy providers, where
    GR00T's local mode ignores a port it never dials. ``"gl"`` opens a native
    window and ``"null"`` is a sink; neither binds anything, so neither has an
    opinion about the port and refusing one there would reject a call that works.
    """

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_the_gl_window_ignores_the_port(self, port: Any) -> None:
        stub = _viewer_stub(has_display=True)
        result = NewtonSimEngine.open_viewer(stub, "gl", port=port)
        assert result["status"] == "success", (port, result)
        assert [kind for kind, _ in stub.built] == ["gl"]

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_the_null_sink_ignores_the_port(self, port: Any) -> None:
        stub = _viewer_stub()
        result = NewtonSimEngine.open_viewer(stub, "null", port=port)
        assert result["status"] == "success", (port, result)
        assert [kind for kind, _ in stub.built] == ["null"]

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_auto_ignores_it_when_a_display_resolves_to_gl(self, port: Any) -> None:
        stub = _viewer_stub(has_display=True)
        assert NewtonSimEngine.open_viewer(stub, "auto", port=port)["status"] == "success"
        assert [kind for kind, _ in stub.built] == ["gl"]


class TestEarlierRefusalsStillPrecedeThePort:
    """The port guard must not reorder the refusals that already existed."""

    def test_no_world_still_wins(self) -> None:
        stub = _viewer_stub()
        stub._world = None
        result = NewtonSimEngine.open_viewer(stub, "viser", port=99999)
        assert result["status"] == "error"
        assert "create_world" in _text(result)

    def test_an_unknown_viewer_kind_still_wins(self) -> None:
        """A typo in ``viewer=`` is the more useful thing to report."""
        result = NewtonSimEngine.open_viewer(_viewer_stub(), "vizer", port=99999)
        assert result["status"] == "error"
        assert "vizer" in _text(result)

    def test_an_already_open_viewer_still_wins(self) -> None:
        stub = _viewer_stub()
        assert NewtonSimEngine.open_viewer(stub, "viser", port=8080)["status"] == "success"
        result = NewtonSimEngine.open_viewer(stub, "viser", port=99999)
        assert result["status"] == "success"
        assert "already open" in _text(result)


# --------------------------------------------------------------------------- #
# Structural guard: a port arriving on a *method* was invisible to every scan   #
# --------------------------------------------------------------------------- #
def _simulation_module_paths() -> list[Path]:
    """Every backend module under ``strands_robots.simulation``."""
    root = Path(inspect.getfile(simulation_pkg)).parent
    return sorted(root.rglob("*.py"))


def _functions_taking_a_port(tree: ast.Module) -> list[ast.FunctionDef]:
    """Functions declaring a parameter named exactly ``port``.

    Exactly ``port`` rather than any ``*_port`` suffix, matching the existing
    provider scan: ``lerobot_teleoperate``'s ``robot_port`` / ``teleop_port`` are
    serial device paths (``/dev/ttyUSB0``), not TCP ports, and a suffix match
    would demand a TCP domain on them.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        if "port" in names:
            found.append(node)
    return found


def _calls_the_shared_domain(func: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tcp_port_error"
        for node in ast.walk(func)
    )


class TestNoSimulationSurfaceShipsAnUnguardedPort:
    """A port on a *method* was structurally invisible before this scan.

    Every port-domain drift guard in the suite enumerates ``__init__``
    definitions - ``TestNoProviderShipsAnUnguardedPort`` over
    ``policies/*/policy.py``, and the Device Connect scan over exported driver
    constructors - because until now every port arrived as a constructor
    parameter. ``open_viewer`` is the one that does not, so no existing scan
    could have reported it and a second backend method taking a port would have
    been just as unreported. This closes that shape for the simulation package.

    Scoped to the simulation backends rather than the whole package on purpose:
    the internal transport clients (``Gr00tInferenceClient`` and its siblings)
    take a ``port`` they deliberately do not re-validate, because their dialing
    provider refuses it first and ``TestRefusalPrecedesTheTransport`` pins that
    ordering. Widening this scan would demand a second, redundant guard there.
    """

    def test_every_simulation_surface_taking_a_port_validates_it(self) -> None:
        offenders = []
        seen = []
        for path in _simulation_module_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for func in _functions_taking_a_port(tree):
                seen.append(func.name)
                if not _calls_the_shared_domain(func):
                    offenders.append(f"{path.name}::{func.name}")
        assert "open_viewer" in seen, seen
        assert offenders == [], (
            "these simulation surfaces accept a port without validating it against "
            f"strands_robots.utils.tcp_port_error: {offenders}"
        )

    def test_the_scan_detects_an_unguarded_method(self) -> None:
        """A planted omission is reported, so an empty result means clean sources."""
        planted = ast.parse(
            "class RogueEngine:\n    def open_dashboard(self, port=8080):\n        return f'http://localhost:{port}'\n"
        )
        functions = _functions_taking_a_port(planted)
        assert [f.name for f in functions] == ["open_dashboard"]
        assert not _calls_the_shared_domain(functions[0])

    def test_the_scan_accepts_a_guarded_method(self) -> None:
        """Non-vacuity in the other direction: a guarded method is not an offender."""
        guarded = ast.parse(
            "class TidyEngine:\n"
            "    def open_dashboard(self, port=8080):\n"
            "        if tcp_port_error(port, 'port', 'TidyEngine'):\n"
            "            return None\n"
        )
        functions = _functions_taking_a_port(guarded)
        assert [f.name for f in functions] == ["open_dashboard"]
        assert _calls_the_shared_domain(functions[0])


class TestTheNeighbouringWindowSizeIsSettledElsewhere:
    """``width`` / ``height`` are a different domain, and it is now applied.

    This class replaces the premise it used to hold - that an unusable ``"gl"``
    window size was accepted and forwarded verbatim - because that axis has since
    been taken up. They are pixel counts, so they belong to
    :func:`~strands_robots.utils.positive_count_error`, the floor this backend's
    ``add_camera`` and render family already share (an integral float is refused
    there rather than coerced, because a dimension is consumed directly as an
    array bound - which is why it is that domain and not the looser
    ``positive_whole_number_error`` used for frame rates).

    The behavioural coverage lives with the rest of that quantity, in
    ``tests/simulation/test_camera_pixel_count_domain.py``, so this file stays
    about the port. What is kept here is the property the two guards share and
    which this module's own measurement established: each is applied only on the
    branch that reads it.
    """

    @pytest.mark.parametrize("bad", (0, -1, 1280.5, NAN, True, "1280", None))
    def test_an_unusable_gl_window_size_is_refused_on_the_branch_that_reads_it(self, bad: Any) -> None:
        stub = _viewer_stub(has_display=True)
        result = NewtonSimEngine.open_viewer(stub, "gl", width=bad, height=bad)
        assert result["status"] == "error", (bad, result)
        assert stub.built == []

    @pytest.mark.parametrize("bad", (0, -1, 1280.5, NAN, True, "1280", None))
    def test_the_viser_branch_ignores_it_and_so_does_not_check_it(self, bad: Any) -> None:
        """The mirror image of the port scoping this module measures.

        ``ViewerViser`` is never handed a width, so refusing one for it would be
        a false rejection - exactly the reason ``port`` is checked only when the
        viser branch is the one that will bind it.
        """
        stub = _viewer_stub(has_display=True)
        result = NewtonSimEngine.open_viewer(stub, "viser", port=8080, width=bad, height=bad)
        assert result["status"] == "success", (bad, result)
        assert stub.built == [("viser", {"port": 8080})]
