"""A render-dimension refusal names the method that was called.

Both the MuJoCo and the Newton backend funnel the width/height domain of every
render surface through one helper - ``_validate_render_dims`` on MuJoCo,
``_resolve_camera_view`` on Newton - and each helper used to spell the subject
of its own refusals as the literal ``render``. It serves more than ``render``,
so a caller of another entry point was told about a method it had not called:

* MuJoCo, five callers: ``render``, ``render_depth``, ``get_frame``,
  ``get_camera_params``, ``add_camera``. Three of them reported ``render``.
* Newton, three callers: ``render``, ``get_frame``, ``get_camera_params``. Two
  of them reported ``render``.

The reader is usually an agent - ``render`` and ``add_camera`` are tool-callable
and return the text in an error envelope, while ``get_frame`` and
``get_camera_params`` raise it - and the repair it suggests is a call the caller
never made. ``render`` also has a *different signature and return type* from
the two raising methods, so following the message means editing an unrelated
call.

That the misattribution mattered was already established in the tree: MuJoCo's
``add_camera`` repaired it after the fact with
``text.replace("render:", "add_camera:", 1)``, a coupling to the literal prefix
of every message the guard can return, which a rewording would silently break.
Isaac never had the defect - it passes ``"get_frame"`` / ``"get_camera_params"``
to ``positive_count_error`` at each call site - and its two call sites are the
control below: the target state is the one a sibling backend already ships, and
the ~20 domain helpers in :mod:`strands_robots.utils` all take their caller's
name for the same reason.

The structural cell is the one that keeps this closed: it derives the expected
subject from each call's own enclosing method, so a sixth entry point cannot
join the funnel without naming itself.
"""

import ast
import inspect
import pathlib
import types
from typing import Any

import pytest

from strands_robots.simulation.newton.simulation import NewtonSimEngine

_ROOT = pathlib.Path(inspect.getfile(NewtonSimEngine)).resolve().parents[3]

# ``(module, shared guard, how many calls to it that module must still hold)``.
# The floors are the populations measured when this was written - four render
# surfaces in MuJoCo's rendering mixin, ``add_camera`` in its engine, and
# Newton's three - and exist so a guard that stops being shared cannot make the
# rule below vacuous.
_FUNNELS = (
    ("strands_robots/simulation/mujoco/rendering.py", "_validate_render_dims", 4),
    ("strands_robots/simulation/mujoco/simulation.py", "_validate_render_dims", 1),
    ("strands_robots/simulation/newton/simulation.py", "_resolve_camera_view", 3),
)


def _context_arguments(path: pathlib.Path, helper: str) -> list[tuple[str, str | None]]:
    """Every call to ``helper``, as ``(enclosing method, context literal)``.

    ``None`` for the context when the last argument is not a string literal,
    which is itself a finding: a subject computed at the call site is not one a
    reader can check against the method it names.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls: list[tuple[str, str | None]] = []
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            # the definition itself is not a call, and neither is a docstring
            if isinstance(func, ast.Attribute) and func.attr == helper and node.args:
                last = node.args[-1]
                literal = last.value if isinstance(last, ast.Constant) and isinstance(last.value, str) else None
                calls.append((stack[-1] if stack else "<module>", literal))
            self.generic_visit(node)

    Visitor().visit(tree)
    return calls


class TestEveryCallerNamesItself:
    """Derived from the tree, so a new entry point joins the rule on arrival."""

    @pytest.mark.parametrize(("module", "helper", "floor"), _FUNNELS)
    def test_the_context_is_the_enclosing_method(self, module: str, helper: str, floor: int) -> None:
        calls = _context_arguments(_ROOT / module, helper)
        assert len(calls) >= floor, f"{module}: found {len(calls)} calls to {helper}, expected at least {floor}"
        wrong = [(method, ctx) for method, ctx in calls if ctx != method]
        assert not wrong, (
            f"{module}: {len(wrong)} call(s) to {helper} name a method other than their own caller: {wrong}. "
            "A refusal must name the entry point the caller invoked."
        )

    def test_isaac_already_passed_its_own_name(self) -> None:
        """The control: Isaac needed no change, and holds either way.

        Its render surfaces call the shared ``positive_count_error`` directly
        with their own names, which is the arrangement the two funnels above
        now reach. A fix that regressed Isaac to a hardcoded subject fails here.
        """
        source = (_ROOT / "strands_robots/simulation/isaac/simulation.py").read_text(encoding="utf-8")
        for entry_point in ("get_frame", "get_camera_params"):
            assert f'positive_count_error(arg, arg_name, "{entry_point}")' in source, entry_point


class TestTheSubjectIsNotPatchedAfterTheFact:
    def test_no_caller_rewrites_the_guards_subject(self) -> None:
        """``add_camera`` used to do this, and it is why the parameter exists.

        Rewriting the subject in the caller couples it to the literal prefix of
        every message the guard can return - four of them here - so a reworded
        message keeps the wrong subject and nothing reports it.
        """
        for module, _helper, _floor in _FUNNELS:
            source = (_ROOT / module).read_text(encoding="utf-8")
            for statement in ast.walk(ast.parse(source)):
                if not isinstance(statement, ast.Call):
                    continue
                func = statement.func
                if isinstance(func, ast.Attribute) and func.attr == "replace" and statement.args:
                    first = statement.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        assert not first.value.startswith("render:"), (
                            f"{module}: a caller still rewrites the refusal subject via "
                            f"str.replace({first.value!r}, ...) instead of passing its own name."
                        )


class TestMujocoEntryPointsNameThemselves:
    """The dimension guard short-circuits before any GL context, so these run headless."""

    @pytest.fixture
    def sim(self) -> Any:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        engine = Simulation()
        engine.create_world()
        yield engine
        engine.destroy()

    @pytest.mark.parametrize("entry_point", ["render", "render_depth"])
    def test_envelope_surfaces(self, sim: Any, entry_point: str) -> None:
        result = getattr(sim, entry_point)(camera_name="default", width=0, height=48)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith(f"{entry_point}: "), text

    def test_add_camera(self, sim: Any) -> None:
        result = sim.add_camera(name="cam", position=[0.5, 0.0, 0.3], target=[0.0, 0.0, 0.0], width=0, height=48)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith("add_camera: "), text

    @pytest.mark.parametrize("entry_point", ["get_frame", "get_camera_params"])
    def test_raising_surfaces(self, sim: Any, entry_point: str) -> None:
        with pytest.raises(ValueError, match=rf"^{entry_point}: "):
            getattr(sim, entry_point)(camera_name="default", width=0, height=48)


class TestNewtonEntryPointsNameThemselves:
    """Driven through the production resolver bound to a stub, as its sibling
    suite does, so the attribution is graded without the Newton runtime."""

    @staticmethod
    def _stub() -> types.SimpleNamespace:
        stub = types.SimpleNamespace(default_width=640, default_height=480, _world=None)
        stub._resolve_camera_view = types.MethodType(NewtonSimEngine._resolve_camera_view, stub)
        return stub

    @pytest.mark.parametrize("entry_point", ["render", "get_frame", "get_camera_params"])
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_each_surface_reports_its_own_name(self, entry_point: str, param: str) -> None:
        stub = self._stub()
        dims: dict[str, int] = {"width": 320, "height": 240}
        dims[param] = 0
        with pytest.raises(ValueError, match=rf"^{entry_point}: {param} "):
            stub._resolve_camera_view("default", dims["width"], dims["height"], entry_point)
