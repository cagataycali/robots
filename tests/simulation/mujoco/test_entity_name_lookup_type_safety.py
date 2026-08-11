"""A MuJoCo entity name that is not a string is reported, not crashed on.

``mujoco.mj_name2id`` declares its name parameter as ``const char *`` and the
pybind11 binding maps Python ``None`` onto a NULL pointer instead of rejecting
it. MuJoCo dereferences that pointer while comparing names, so the call does not
raise - it terminates the interpreter with SIGSEGV. Five agent-callable methods
routed a caller-supplied name straight into that binding, so a single argument of
the wrong type killed the process::

    get_body_state(body_name=None)        SIGSEGV
    set_body_properties(body_name=None)   SIGSEGV
    apply_force(body_name=None)           SIGSEGV
    attach_bodies(parent=None)            SIGSEGV
    set_joint_positions({None: 0.1})      SIGSEGV

Nothing above the call could recover: the agent-tool envelope, the caller's
``except`` clauses, and any open recording all died with the process. Three more
methods reached the shared "did you mean" reporter instead, where
``difflib.get_close_matches(None, ...)`` raised a bare ``TypeError`` past the
envelope those methods document as their only failure channel.

These tests pin that every one of those names now resolves to "not found" and is
reported through the envelope, that the reporter can render a non-string name,
and - by walking the AST - that no module reaches the binding directly, so a
lookup added later inherits the guard instead of re-opening the crash.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation import Simulation  # noqa: E402
from strands_robots.simulation.mujoco.backend import mj_name_to_id  # noqa: E402
from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402

# Names that cannot identify an entity. ``None`` is the one an agent produces by
# omitting a value it believes optional; the others cover the neighbouring scalar
# types. All are hashable, which keeps the probe on the tables this change owns:
# an unhashable name (``["crate"]``) is refused earlier by the registry's own
# ``name in dict`` membership test raising ``TypeError: unhashable type``, which
# is a separate mechanism with its own call sites and is not in scope here.
NON_STRING_NAMES: tuple[Any, ...] = (None, 5, 2.5, True, b"crate")


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_entity_name_types", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _seeded_world(sim) -> None:
    """Compile a world holding one free-floating body named ``crate``.

    ``sim`` is left un-annotated: ``Simulation`` is a lazy module re-export, so
    annotating it makes mypy read every ``sim._world`` access as ``None``.

    The body matters. Every assertion below is that a *non-string* name is
    refused, which would also hold vacuously in an empty world - the reporter
    short-circuits when the model has nothing to suggest, and the crash sites
    are only reachable once there is a name table to compare against.
    """
    assert sim.create_world()["status"] == "success"
    added = sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.0, 0.0, 0.3])
    assert added["status"] == "success"


def _text(result: dict[str, Any]) -> str:
    return "".join(block.get("text", "") for block in result.get("content", []))


class TestTheBindingIsGenuinelyUnsafe:
    """The hazard the wrapper exists for, measured rather than asserted."""

    def test_the_raw_binding_does_not_survive_a_none_name(self) -> None:
        """Calling ``mj_name2id`` with ``None`` never returns normally.

        Run out-of-process because the failure mode under test is fatal: an
        in-process call would take the test session down with it. The assertion
        is deliberately "did not complete", not "crashed with SIGSEGV", so it
        still holds if a later mujoco starts rejecting the argument instead -
        either way the wrapper is what turns it into a reportable miss.
        """
        source = (
            "import mujoco as mj\n"
            "m = mj.MjModel.from_xml_string("
            '\'<mujoco><worldbody><body name="a"><geom size="0.1"/></body></worldbody></mujoco>\')\n'
            "mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, None)\n"
            "print('RETURNED')\n"
        )
        done = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=180)
        assert "RETURNED" not in done.stdout, "mujoco accepted a None name and returned; wrapper premise stale"
        assert done.returncode != 0


class TestTheWrapperResolvesToNotFound:
    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_a_non_string_name_resolves_to_minus_one(self, name: Any) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="a"><geom size="0.1"/></body></worldbody></mujoco>'
        )
        assert mj_name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1

    def test_a_string_name_still_resolves(self) -> None:
        """The wrapper is transparent for the names that do identify an entity."""
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="a"><geom size="0.1"/></body></worldbody></mujoco>'
        )
        assert mj_name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "a") == 1
        assert mj_name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "nosuch") == -1


class TestTheCrashSitesReportInstead:
    """Each of these terminated the process with SIGSEGV before the guard."""

    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_get_body_state(self, sim, name: Any) -> None:
        _seeded_world(sim)
        result = sim.get_body_state(body_name=name)
        assert result["status"] == "error"
        assert "not found" in _text(result)

    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_set_body_properties(self, sim, name: Any) -> None:
        _seeded_world(sim)
        result = sim.set_body_properties(body_name=name, mass=1.0)
        assert result["status"] == "error"
        assert "not found" in _text(result)

    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_apply_force(self, sim, name: Any) -> None:
        _seeded_world(sim)
        result = sim.apply_force(body_name=name, force=[1.0, 0.0, 0.0])
        assert result["status"] == "error"
        assert "not found" in _text(result)

    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_attach_bodies(self, sim, name: Any) -> None:
        _seeded_world(sim)
        result = sim.attach_bodies(parent=name, child="crate")
        assert result["status"] == "error"
        assert "not found" in _text(result)

    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_set_joint_positions_dict_key(self, sim, name: Any) -> None:
        """The name arrives as a mapping KEY on this method rather than a value."""
        _seeded_world(sim)
        result = sim.set_joint_positions({name: 0.1})
        assert result["status"] == "error"
        assert "not joints in this model" in _text(result)


class TestTheReporterCanRenderANonStringName:
    """These three reached the shared "did you mean" block and raised there."""

    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_move_object(self, sim, name: Any) -> None:
        _seeded_world(sim)
        result = sim.move_object(name=name, position=[0.2, 0.0, 0.3])
        assert result["status"] == "error"
        assert "not found" in _text(result)

    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_remove_object(self, sim, name: Any) -> None:
        _seeded_world(sim)
        result = sim.remove_object(name=name)
        assert result["status"] == "error"
        assert "not found" in _text(result)

    @pytest.mark.parametrize("name", NON_STRING_NAMES, ids=repr)
    def test_remove_camera(self, sim, name: Any) -> None:
        _seeded_world(sim)
        result = sim.remove_camera(name=name)
        assert result["status"] == "error"
        assert "not found" in _text(result)

    def test_the_suggestion_is_still_offered_for_a_string_typo(self, sim) -> None:
        """Guarding the reporter must not cost the close-match it exists for."""
        _seeded_world(sim)
        result = sim.get_body_state(body_name="crat")
        assert result["status"] == "error"
        assert "Did you mean" in _text(result)
        assert "crate" in _text(result)


class TestTheSessionSurvives:
    def test_the_world_is_still_usable_after_a_refused_lookup(self, sim) -> None:
        """The point of the guard: a bad name costs one error, not the session.

        Pre-fix there was nothing to assert here - the interpreter was gone.
        """
        _seeded_world(sim)
        assert sim.get_body_state(body_name=None)["status"] == "error"
        assert sim.apply_force(body_name=None, force=[1.0, 0.0, 0.0])["status"] == "error"
        assert sim.get_body_state(body_name="crate")["status"] == "success"
        assert sim.step(n_steps=5)["status"] == "success"

    @requires_gl
    def test_the_world_still_renders_after_a_refused_lookup(self, sim) -> None:
        """Rendering is the one liveness check that needs a host GL context.

        Split from the case above so the assertions that need no GL keep running
        on a headless host without EGL/OSMesa. Left inline, ``render`` returns
        ``{"status": "error"}`` there for a reason unrelated to entity-name
        lookups, and the whole liveness pin was reported as a failure.
        """
        _seeded_world(sim)
        assert sim.get_body_state(body_name=None)["status"] == "error"
        assert sim.render(camera_name="default")["status"] == "success"


def _mujoco_backend_dir() -> Path:
    """Locate the backend package from a symbol, not from a path literal."""
    from strands_robots.simulation.mujoco import backend

    return Path(inspect.getfile(backend)).parent


def _direct_binding_calls(source: str) -> list[str]:
    """Every ``<mod>.mj_name2id(...)`` call in *source*, as ``line:attr`` labels."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "mj_name2id":
            found.append(f"{node.lineno}:{node.func.attr}")
    return found


class TestNoModuleReachesTheBindingDirectly:
    """The guard is only complete while every lookup goes through the wrapper."""

    def test_only_the_wrapper_calls_the_binding(self) -> None:
        backend_dir = _mujoco_backend_dir()
        offenders = {}
        for path in sorted(backend_dir.glob("*.py")):
            if path.name == "backend.py":
                continue  # defines the wrapper; its own call is the sanctioned one
            calls = _direct_binding_calls(path.read_text(encoding="utf-8"))
            if calls:
                offenders[path.name] = calls
        assert not offenders, (
            f"these modules call mujoco's name binding directly: {offenders}. "
            "Use mj_name_to_id from .backend so a non-string name cannot crash the process."
        )

    def test_the_scanner_detects_a_planted_call(self) -> None:
        """A scanner that silently matched nothing would look like a clean suite."""
        planted = "def f(model, mj, name):\n    return mj.mj_name2id(model, 1, name)\n"
        assert _direct_binding_calls(planted) == ["2:mj_name2id"]
        assert _direct_binding_calls("def f():\n    return 1\n") == []

    def test_the_wrapper_module_still_owns_exactly_one_call(self) -> None:
        """Locates the one sanctioned call site, so the exemption cannot go stale."""
        backend_source = (_mujoco_backend_dir() / "backend.py").read_text(encoding="utf-8")
        assert len(_direct_binding_calls(backend_source)) == 1
