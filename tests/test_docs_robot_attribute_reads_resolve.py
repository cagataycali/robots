"""Every documented attribute read on a ``Robot()`` result must exist on what it returns.

``Robot()`` is polymorphic. ``mode="sim"`` builds a
:class:`~strands_robots.simulation.Simulation`; ``mode="real"`` builds a
:class:`strands_robots.hardware_robot.Robot` wrapper *unless* the robot's
registry entry declares ``hardware.driver = "strands"``, in which case the
factory returns the native driver itself, whose surface is
:data:`~strands_robots.drivers.DRIVER_SURFACE`. So ``robot.attach_teleop(...)``
is a correct line for a lerobot-backed robot and an :class:`AttributeError` for
a natively-driven one, and the two are spelled identically at the call site.

Neither existing docs grader can see that class of error:

* ``tests/test_docs_real_mode_invocations.py`` grades the robot *name* and the
  *keywords* inside the ``Robot(...)`` call.
* ``tests/test_docs_python_examples_are_callable.py`` grades keyword sets
  against signatures, and its ``_accepted_keywords`` returns ``None`` ("any
  keyword binds") for a callee carrying ``**kwargs`` - which ``Robot`` does.

Attribute access on the factory's *return value* is outside both, so a
documented read of a name the returned object does not carry renders verbatim in
the docs and raises on the first line a reader copies. This module closes that
gap: it resolves each documented read to the type the factory would return for
that ``(name, mode)`` pair and requires the name to exist there.

The surface a read is graded against is the union of three sources, because an
attribute can arrive from any of them:

* the class and its MRO, via :func:`dir`;
* ``self.X = ...`` assigned anywhere in the MRO - instance state a class sets up
  in ``__init__`` or a lazy initialiser, which :func:`dir` on the class cannot
  see;
* ``instance.X = ...`` bound by the factory in :mod:`strands_robots.robot`.
  ``run``, ``mesh`` and ``peer_id`` are bound there and appear on no class, so
  omitting this third source reports three offenders that are not defects.

Because the documented corpus is expected to be clean, the rule is also graded
over constructed exemplars rather than relying on the corpus to exercise its
own failing branch.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

import pytest

import strands_robots
import strands_robots.hardware_robot as hardware_robot
import strands_robots.robot as robot_factory
from strands_robots.drivers import get_native_driver_class
from strands_robots.registry import get_robot, resolve_name

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: Modes whose returned type is a simulation rather than a hardware surface.
_SIM_MODES = frozenset({None, "sim"})


def _docs_sources() -> list[Path]:
    """Every documentation file whose ``python`` fences are graded."""
    files = sorted(_REPO_ROOT.glob("docs/**/*.md")) + [_REPO_ROOT / "README.md"]
    return [path for path in files if path.exists()]


def _runnable(block: str) -> str:
    """Return the parseable source of a fence, dropping doctest output lines."""
    if ">>> " in block:
        return "\n".join(line[4:] for line in block.splitlines() if line.startswith((">>> ", "... ")))
    return block


def _instance_surface(cls: type) -> set[str]:
    """Names reachable on an instance of ``cls``.

    ``dir`` misses instance state, so every ``self.X = ...`` target anywhere in
    the MRO is credited as well.
    """
    names = set(dir(cls))
    for klass in cls.__mro__:
        try:
            source = textwrap.dedent(inspect.getsource(klass))
        except (OSError, TypeError):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Store)
            ):
                names.add(node.attr)
    return names


def _factory_bound_names() -> set[str]:
    """Attributes the ``Robot()`` factory binds onto the instance it returns.

    ``run`` is bound here rather than defined on any class, so a surface derived
    from the classes alone would report every documented ``.run()`` as missing.
    """
    tree = ast.parse(inspect.getsource(robot_factory))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def _declares_a_native_driver(name: str) -> bool:
    """Whether the registry entry for ``name`` asks for its native driver."""
    record = get_robot(resolve_name(name)) or {}
    return ((record.get("hardware") or {}).get("driver")) == "strands"


def _native_driver_class(name: str) -> type:
    """The driver class the registry names for ``name``.

    ``get_native_driver_class`` answers ``None`` for a robot with no native
    driver, which :func:`_declares_a_native_driver` has already ruled out.
    """
    driver = get_native_driver_class(resolve_name(name))
    assert driver is not None, f"{name!r} declares hardware.driver='strands' but registers no driver class"
    return driver


def _hardware_surfaces(name: str, mode: str | None) -> dict[str, set[str]]:
    """Surfaces ``Robot(name, mode=mode)`` can return, for a non-sim mode."""
    factory = _factory_bound_names()
    if mode == "real" and _declares_a_native_driver(name):
        driver = _native_driver_class(name)
        return {driver.__name__: _instance_surface(driver) | factory}
    wrapper = {"HardwareRobot": _instance_surface(hardware_robot.Robot) | factory}
    if mode == "real":
        return wrapper
    # "auto" resolves at runtime from the environment, so either is acceptable.
    return wrapper | {"Simulation": _simulation_surface()}


def _simulation_surface() -> set[str]:
    """The surface of the simulation the factory builds, or an empty set."""
    from strands_robots.simulation import Simulation

    return _instance_surface(Simulation) | _factory_bound_names()


def _documented_reads(source: str, origin: str) -> list[tuple[str, str, str | None, str]]:
    """Return ``(origin, robot_name, mode, attribute)`` for each read in ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    def _call_target(call: ast.Call) -> tuple[str | None, str | None]:
        keywords = {kw.arg: getattr(kw.value, "value", None) for kw in call.keywords}
        name = call.args[0].value if call.args and isinstance(call.args[0], ast.Constant) else keywords.get("name")
        return (name if isinstance(name, str) else None), keywords.get("mode")

    bound: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "Robot"
        ):
            name, mode = _call_target(node.value)
            if name is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = (name, mode)

    reads: list[tuple[str, str, str | None, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id in bound:
            name, mode = bound[base.id]
        elif isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "Robot":
            resolved, mode = _call_target(base)
            if resolved is None:
                continue
            name = resolved
        else:
            continue
        reads.append((origin, name, mode, node.attr))
    return reads


def _all_documented_reads() -> list[tuple[str, str, str | None, str]]:
    """Every distinct attribute read on a documented ``Robot()`` result."""
    reads: list[tuple[str, str, str | None, str]] = []
    for path in _docs_sources():
        origin = str(path.relative_to(_REPO_ROOT))
        for match in _PYTHON_FENCE.finditer(path.read_text(encoding="utf-8")):
            reads.extend(_documented_reads(_runnable(match.group(1)), origin))
    seen: set[tuple[str, str, str | None, str]] = set()
    distinct: list[tuple[str, str, str | None, str]] = []
    for read in reads:
        if read not in seen:
            seen.add(read)
            distinct.append(read)
    return distinct


def _unresolved(reads: list[tuple[str, str, str | None, str]]) -> list[str]:
    """Return a report line for each read the returned type cannot answer."""
    offenders: list[str] = []
    for origin, name, mode, attribute in reads:
        try:
            resolve_name(name)
        except ValueError:
            # An unregistered name is graded by test_docs_real_mode_invocations.
            continue
        surfaces = _simulation_surfaces() if mode in _SIM_MODES else _hardware_surfaces(name, mode)
        if not any(attribute in surface for surface in surfaces.values()):
            offenders.append(f"{origin}: Robot({name!r}, mode={mode!r}).{attribute} is not on {sorted(surfaces)}")
    return offenders


def _simulation_surfaces() -> dict[str, set[str]]:
    """The sim-mode surface, keyed for reporting."""
    return {"Simulation": _simulation_surface()}


#: Pages that must contribute a read in each partition, set below today's
#: counts (5 hardware-mode, 20 sim-mode) so an ordinary docs edit does not trip
#: it while a narrowed file glob does.
_MINIMUM_HARDWARE_SOURCES = 4
_MINIMUM_SIMULATION_SOURCES = 10


def _assert_the_scan_reaches_the_docs_tree(reads: list[tuple[str, str, str | None, str]], minimum: int) -> None:
    """Refuse a corpus small enough that the guard could be passing vacuously."""
    assert reads, "no documented attribute read was found - the scan stopped seeing the docs"
    sources = {read[0] for read in reads}
    under_docs = {source for source in sources if source.startswith("docs/")}
    assert len(sources) >= minimum, (
        f"only {sorted(sources)} contributed a read, fewer than the {minimum} expected; "
        "a narrowed file glob would leave the rest of the documentation ungraded"
    )
    assert under_docs, f"no page under docs/ contributed a read, only {sorted(sources)}"


class TestEveryDocumentedReadResolves:
    """A documented read must name something the returned object carries."""

    def test_no_hardware_mode_read_is_unresolvable(self) -> None:
        reads = [read for read in _all_documented_reads() if read[2] not in _SIM_MODES]
        _assert_the_scan_reaches_the_docs_tree(reads, _MINIMUM_HARDWARE_SOURCES)
        offenders = _unresolved(reads)
        assert offenders == [], "documented reads that raise AttributeError as written:\n  " + "\n  ".join(offenders)

    def test_no_simulation_read_is_unresolvable(self) -> None:
        pytest.importorskip("mujoco", reason="the simulation surface needs the [sim-mujoco] extra")
        reads = [read for read in _all_documented_reads() if read[2] in _SIM_MODES]
        _assert_the_scan_reaches_the_docs_tree(reads, _MINIMUM_SIMULATION_SOURCES)
        offenders = _unresolved(reads)
        assert offenders == [], "documented reads that raise AttributeError as written:\n  " + "\n  ".join(offenders)


class TestThePolymorphismThisGrades:
    """The premise: which surface a read is graded against depends on the robot."""

    def test_a_robot_declaring_a_native_driver_returns_the_driver_surface(self) -> None:
        natively_driven = [name for name in ("unitree_g1",) if _declares_a_native_driver(name)]
        assert natively_driven, "no registry entry declares hardware.driver='strands' - the premise is gone"
        surfaces = _hardware_surfaces(natively_driven[0], "real")
        assert "HardwareRobot" not in surfaces, "a natively-driven robot must not be graded as the lerobot wrapper"

    def test_the_two_hardware_surfaces_disagree_about_a_real_method(self) -> None:
        """``attach_teleop`` exists on one surface and not the other, which is the whole risk."""
        wrapper = _instance_surface(hardware_robot.Robot)
        driver = _instance_surface(_native_driver_class("unitree_g1"))
        assert "attach_teleop" in wrapper
        assert "attach_teleop" not in driver

    def test_a_factory_bound_attribute_is_credited(self) -> None:
        """``run`` is bound onto the instance and is on no class in the MRO."""
        bound = _factory_bound_names()
        assert "run" in bound
        assert not hasattr(hardware_robot.Robot, "run")
        assert "run" in _hardware_surfaces("so100", "real")["HardwareRobot"]


class TestTheRuleIsGradedOnConstructedExemplars:
    """The documented corpus is expected to be clean, so the rule is graded directly."""

    _ACCEPTED = 'r = Robot("so100", mode="real")\nr.attach_teleop("so101_leader")\n'
    _MISSPELLED = 'r = Robot("so100", mode="real")\nr.attach_teleoperator("so101_leader")\n'
    _WRONG_SURFACE = 'r = Robot("unitree_g1", mode="real")\nr.attach_teleop("so101_leader")\n'
    _FACTORY_BOUND = 'r = Robot("so100", mode="real")\nr.run()\n'
    # control_frequency is assigned as self.X and is on no class, so this
    # exemplar is answered only by the MRO-assignment source.
    _INSTANCE_STATE = 'r = Robot("so100", mode="real")\nr.control_frequency\n'

    @pytest.mark.parametrize(
        ("exemplar", "expected"),
        [
            (_ACCEPTED, False),
            (_FACTORY_BOUND, False),
            (_INSTANCE_STATE, False),
            (_MISSPELLED, True),
            (_WRONG_SURFACE, True),
        ],
        ids=["accepted", "factory-bound", "instance-state", "misspelled", "wrong-surface"],
    )
    def test_the_rule_separates_these(self, exemplar: str, expected: bool) -> None:
        offenders = _unresolved(_documented_reads(exemplar, "exemplar.md"))
        assert bool(offenders) is expected, f"exemplar graded {bool(offenders)}, expected {expected}: {offenders}"

    def test_the_exemplars_reach_both_verdicts(self) -> None:
        outcomes = {
            bool(_unresolved(_documented_reads(exemplar, "exemplar.md")))
            for exemplar in (
                self._ACCEPTED,
                self._FACTORY_BOUND,
                self._INSTANCE_STATE,
                self._MISSPELLED,
                self._WRONG_SURFACE,
            )
        }
        assert outcomes == {True, False}, f"the exemplars only ever produce {outcomes}"
