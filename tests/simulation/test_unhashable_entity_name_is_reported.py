"""A registry lookup reports any entity name instead of raising on it.

Every simulation entity is addressed by name, and each backend resolves a
caller-supplied name against a name-keyed registry before it does anything
else. A bare ``name in registry`` / ``registry.get(name)`` is not total: for a
name that is not hashable (a list, a dict, a set) the lookup itself raises
``TypeError: unhashable type``, so the unknown-entity error path that the
lookup guards is never reached and the exception escapes the agent-tool dict
that the surrounding method documents as its only failure channel.

These tests pin the lookup as total on every backend that keeps such a
registry: a name of any type resolves to a verdict, a name that cannot be a key
resolves to "absent", and the message the method already had reports it. The AST
check at the end is what keeps a lookup added later inside that rule.
"""

from __future__ import annotations

import ast
import inspect
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.models import SimRobot, SimWorld, registered, registry_entry

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A name that cannot key a registry, one per unhashable builtin a caller might
# plausibly pass by mistake (a single-element list is what wrapping a name in
# brackets produces; a dict is what a half-built kwargs mapping looks like).
UNHASHABLE: list[tuple[str, Any]] = [
    ("list", ["crate"]),
    ("dict", {"crate": 1}),
    ("set", {"crate"}),
    ("bytearray", bytearray(b"crate")),
]

# One hinge with a position servo: enough for a registered robot with joints and
# actuators, and it needs no downloaded asset so this runs anywhere.
ARM_XML = """
<mujoco model="probe_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1" range="-1.5 1.5" limited="true" damping="1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator><position name="pan_act" joint="pan" kp="20" ctrlrange="-1.5 1.5"/></actuator>
</mujoco>
"""


@pytest.fixture
def sim(tmp_path):
    xml = tmp_path / "arm.xml"
    xml.write_text(ARM_XML)
    s = Simulation(tool_name="unhashable_name_sim", mesh=False)
    s.create_world()
    s.add_object("crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.3, 0.0, 0.05], is_static=False)
    s.add_robot(name="arm", urdf_path=str(xml))
    s.add_camera(name="look", position=[0.9, -0.9, 0.6], target=[0.3, 0.0, 0.1])
    # The refusals below are only meaningful against a world that does hold the
    # entities being asked for, so pin the premise rather than assuming it.
    assert "crate" in s._world.objects
    assert "arm" in s._world.robots
    assert "look" in s._world.cameras
    yield s
    s.cleanup()


def _mujoco_lookups(s: Simulation) -> dict[str, Any]:
    """Every MuJoCo entry point that resolves a caller-supplied name."""
    return {
        "move_object": lambda n: s.move_object(name=n, position=[0.2, 0.0, 0.3]),
        "remove_object": lambda n: s.remove_object(name=n),
        "remove_camera": lambda n: s.remove_camera(name=n),
        "remove_robot": lambda n: s.remove_robot(name=n),
        "send_action": lambda n: s.send_action({"pan_act": 0.1}, robot_name=n),
        "get_robot_state": lambda n: s.get_robot_state(robot_name=n),
        "list_bodies": lambda n: s.list_bodies(robot_name=n),
        "get_features": lambda n: s.get_features(robot_name=n),
        "start_policy": lambda n: s.start_policy(robot_name=n, duration=0.2),
        "stop_policy": lambda n: s.stop_policy(robot_name=n),
        "move_to": lambda n: s.move_to(robot_name=n, position=[0.2, 0.0, 0.2]),
        "rotate_wrist": lambda n: s.rotate_wrist(robot_name=n, target_yaw=0.2),
        "actuate_robot": lambda n: s.actuate_robot(robot_name=n),
        "run_policy": lambda n: s.run_policy(robot_name=n, n_steps=2),
        "eval_policy": lambda n: s.eval_policy(robot_name=n, n_episodes=1, max_steps=2),
        "render": lambda n: s.render(camera_name=n, width=64, height=48),
        "render_depth": lambda n: s.render_depth(camera_name=n, width=64, height=48),
    }


def _envelope_miss(call: Callable[[Any], Any], name: Any) -> str | None:
    """Describe how ``call`` failed to report ``name``, or ``None`` if it reported.

    An exception is reported rather than propagated, so one run names every
    method that escaped instead of stopping at the first. That reporting is
    sound only for an exception the method leaked, which is where the caught set
    stops: the escape under test is a ``TypeError`` out of a partial lookup, and
    every type these methods can leak is an ``Exception``.

    It deliberately does not sit at ``BaseException``. A pytest outcome
    (``Skipped`` / ``Failed``) or an operator interrupt (``KeyboardInterrupt`` /
    ``SystemExit``) derives from ``BaseException`` without deriving from
    ``Exception``, and none of them is the API's to leak - absorbing them would
    turn an interrupted run into a verdict string and report a skipped
    dependency as an envelope escape. :class:`TestTheEnvelopeProbe` pins the
    boundary in both directions.
    """
    try:
        result = call(name)
    except Exception as exc:  # noqa: BLE001 - the escape is what is under test
        return f"raised {type(exc).__name__}: {exc}"
    if not isinstance(result, dict) or result.get("status") != "error":
        return f"expected an error envelope, got {result!r}"
    return None


@pytest.mark.parametrize(("label", "name"), UNHASHABLE, ids=[lbl for lbl, _ in UNHASHABLE])
def test_an_unhashable_name_is_reported_as_absent(sim, label, name):
    """Each name-taking method returns its error envelope, not a TypeError."""
    failures = {
        method: miss
        for method, call in _mujoco_lookups(sim).items()
        if (miss := _envelope_miss(call, name)) is not None
    }
    assert not failures, f"a {label} name escaped or was accepted:\n" + "\n".join(
        f"  {k}: {v}" for k, v in sorted(failures.items())
    )


class TestTheEnvelopeProbe:
    """The probe the sweep above shares must not absorb control flow.

    :func:`_envelope_miss` turns one call into a miss description or ``None`` so
    that a single assertion can name every method that escaped. Reporting an
    exception is what makes that message useful, and it has to stay bounded to
    exceptions the API leaked: a probe that also caught pytest's own outcomes
    would report a skipped optional dependency as an envelope escape, and an
    interrupted run as an answer.
    """

    def test_an_error_envelope_is_what_the_probe_asks_for(self):
        assert _envelope_miss(lambda n: {"status": "error"}, ["crate"]) is None

    @pytest.mark.parametrize("accepted", [{"status": "success"}, None, "ok"])
    def test_anything_other_than_an_error_envelope_is_a_miss(self, accepted):
        """A name that cannot address an entity must not be reported as resolved."""
        miss = _envelope_miss(lambda n: accepted, ["crate"])
        assert miss is not None and "expected an error envelope" in miss

    @pytest.mark.parametrize("leaked", [TypeError, ValueError, KeyError])
    def test_an_exception_the_method_leaked_is_reported(self, leaked):
        """The escape under test is a leaked ``TypeError``; it must stay reported.

        Reporting rather than propagating is what lets one run list every method
        that escaped, so it has to survive the narrowing of the caught set.
        """

        def leak(name: Any) -> dict[str, Any]:
            raise leaked("unhashable type: 'list'")

        miss = _envelope_miss(leak, ["crate"])
        assert miss is not None and miss.startswith(f"raised {leaked.__name__}")

    @pytest.mark.parametrize(
        "control_flow",
        [pytest.skip.Exception, pytest.fail.Exception, KeyboardInterrupt, SystemExit],
    )
    def test_control_flow_is_not_absorbed(self, control_flow):
        """A pytest outcome or an operator interrupt passes straight through.

        All four derive from ``BaseException`` without deriving from
        ``Exception``, and none is something a lookup leaked. Absorbing them
        would turn a ``Ctrl-C`` into ``"raised KeyboardInterrupt"`` and then
        assert on it as though the lookup had answered.
        """

        def interrupt(name: Any) -> dict[str, Any]:
            raise control_flow("not the API's to leak")

        with pytest.raises(control_flow):
            _envelope_miss(interrupt, ["crate"])


@pytest.mark.parametrize(("label", "name"), UNHASHABLE, ids=[lbl for lbl, _ in UNHASHABLE])
def test_a_best_effort_lookup_reports_nothing_rather_than_raising(sim, label, name):
    """The two lookups documented as best-effort keep returning an empty list."""
    assert sim.robot_joint_names(name) == []
    assert sim.robot_action_keys(name) == []
    # get_observation has no error channel either; it must still not raise.
    assert isinstance(sim.get_observation(robot_name=name), dict)


def test_the_reported_message_names_the_value_and_what_exists(sim):
    """The existing unknown-entity message is reused, so the caller can recover."""
    text = sim.move_object(name=["crate"], position=[0.2, 0.0, 0.3])["content"][0]["text"]
    assert "not found" in text
    assert "crate" in text  # the available objects are still listed


def test_a_registered_name_is_unaffected(sim):
    """Guarding the lookup must not cost the lookups that resolve."""
    assert sim.move_object(name="crate", position=[0.25, 0.0, 0.2])["status"] == "success"
    assert sim.get_robot_state(robot_name="arm")["status"] == "success"
    assert sim.render(camera_name="look", width=64, height=48)["status"] == "success"


def test_an_unknown_string_name_is_unaffected(sim):
    """A misspelled name still reports the same way it always did."""
    text = sim.move_object(name="crat", position=[0.2, 0.0, 0.3])["content"][0]["text"]
    assert "Object 'crat' not found" in text
    assert "crate" in text


# --------------------------------------------------------------------------- #
# the shared rule                                                             #
# --------------------------------------------------------------------------- #


def test_registered_answers_for_a_name_of_any_type():
    registry = {"crate": object()}
    assert registered(registry, "crate") is True
    assert registered(registry, "nosuch") is False
    for _label, name in UNHASHABLE:
        assert registered(registry, name) is False
    assert registered(registry, None) is False
    assert registered(registry, 3) is False


def test_registry_entry_returns_the_entry_or_none():
    entry = object()
    registry = {"crate": entry}
    assert registry_entry(registry, "crate") is entry
    assert registry_entry(registry, "nosuch") is None
    for _label, name in UNHASHABLE:
        assert registry_entry(registry, name) is None
    assert registry_entry(registry, None) is None


# --------------------------------------------------------------------------- #
# Newton keeps the same registries, so it answers the same way                #
# --------------------------------------------------------------------------- #


def _newton_engine():
    """A Newton engine holding one registered robot, without the newton package.

    Only the registry and the lookup are under test, so the parts of the engine
    that need the simulator are never reached.
    """
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._lock = threading.Lock()
    engine._model = object()  # only its presence is read by the lookups under test
    engine._world = SimWorld()
    engine._world.robots["arm"] = SimRobot(name="arm", urdf_path="", joint_names=["pan"])
    return engine


@pytest.mark.parametrize(("label", "name"), UNHASHABLE, ids=[lbl for lbl, _ in UNHASHABLE])
def test_newton_reports_an_unhashable_name_the_same_way(label, name):
    engine = _newton_engine()
    for method in ("remove_robot", "get_robot_state", "list_bodies", "get_features"):
        result = getattr(engine, method)(name)
        assert isinstance(result, dict) and result.get("status") == "error", (
            f"newton {method} accepted a {label} name: {result!r}"
        )


def test_newton_still_resolves_a_registered_name():
    """A registered name gets past the lookup on Newton too.

    Each of these methods goes on to read simulator state that this skeleton
    does not build, so failing on that state is itself the evidence that the
    name resolved: a name the lookup rejected would have returned the
    unknown-robot envelope instead, without reaching it.
    """
    for method in ("get_robot_state", "list_bodies", "get_features", "remove_robot"):
        with pytest.raises(AttributeError):
            getattr(_newton_engine(), method)("arm")


# --------------------------------------------------------------------------- #
# Isaac keeps its registries on the engine, and answers the same way           #
# --------------------------------------------------------------------------- #


def _isaac_engine() -> Any:
    """An Isaac engine holding one entity of each kind, without Isaac Sim.

    Isaac does not mirror its entities onto :class:`SimWorld`; ``_robots``,
    ``_objects`` and ``_cameras`` on the engine are the registries a name is
    resolved against. Only that resolution is under test, so the simulator-backed
    state each method goes on to read is left unbuilt - a call that gets past the
    lookup fails on that state instead, which is how the tests below tell a name
    that resolved from one that did not.
    """
    from strands_robots.simulation.isaac.config import IsaacConfig
    from strands_robots.simulation.isaac.simulation import (
        IsaacSimulation,
        _CameraState,
        _ObjectState,
        _RobotState,
    )

    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world_created = True
    engine._config = IsaacConfig()
    engine._robots = {"arm": _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["pan"])}
    engine._objects = {
        "crate": _ObjectState(name="crate", prim_path="/World/Objects/crate", shape="box", is_static=False)
    }
    engine._cameras = {"look": _CameraState(name="look", prim_path="/World/Cameras/look", width=64, height=48)}
    engine._prim_registry = ["/World/Robots/arm", "/World/Objects/crate", "/World/Cameras/look"]
    # No Isaac ``World``: the prim deletion each ``remove_*`` attempts is
    # best-effort and reports through the same envelope, so its absence does
    # not stand in the way of the name resolution under test.
    engine._world = None
    engine._action_controllers = {}
    engine._cam_out_size = {}
    engine._cams_rec_state = None
    engine._main_tid = threading.get_ident()
    return engine


# Every Isaac entry point that resolves a caller-supplied name and reports
# through the agent-tool envelope, with the kind of entity each one addresses.
_ISAAC_ENVELOPE_LOOKUPS: dict[str, Callable[[Any, Any], Any]] = {
    "remove_robot": lambda e, n: e.remove_robot(n),
    "remove_object": lambda e, n: e.remove_object(n),
    "remove_camera": lambda e, n: e.remove_camera(n),
    "send_action": lambda e, n: e.send_action({"pan": 0.1}, robot_name=n),
    "get_jacobian": lambda e, n: e.get_jacobian(robot_name=n),
    "move_object": lambda e, n: e.move_object(name=n, position=[0.2, 0.0, 0.3]),
    "set_object_kinematic": lambda e, n: e.set_object_kinematic(n, True),
    "set_object_collision": lambda e, n: e.set_object_collision(n, True),
    "set_robot_pose": lambda e, n: e.set_robot_pose(robot_name=n, position=[0.0, 0.0, 0.0]),
    "set_joint_positions": lambda e, n: e.set_joint_positions({"pan": 0.1}, robot_name=n),
    "install_action_controller": lambda e, n: e.install_action_controller(n, object()),
    "start_cameras_recording": lambda e, n: e.start_cameras_recording(output_dir="/tmp/isaac-name", cameras=[n]),
    "get_body_state": lambda e, n: e.get_body_state(body_name=n),
}


@pytest.mark.parametrize(("label", "name"), UNHASHABLE, ids=[lbl for lbl, _ in UNHASHABLE])
def test_isaac_reports_an_unhashable_name_the_same_way(label, name):
    """Each Isaac method returns its error envelope, not a TypeError."""
    failures = {
        method: miss
        for method, call in _ISAAC_ENVELOPE_LOOKUPS.items()
        if (miss := _envelope_miss(lambda n, _c=call: _c(_isaac_engine(), n), name)) is not None
    }
    assert not failures, f"a {label} name escaped or was accepted on Isaac:\n" + "\n".join(
        f"  {k}: {v}" for k, v in sorted(failures.items())
    )


@pytest.mark.parametrize(("label", "name"), UNHASHABLE, ids=[lbl for lbl, _ in UNHASHABLE])
def test_isaac_best_effort_lookups_report_nothing_rather_than_raising(label, name):
    """The two Isaac lookups with no error channel keep answering empty."""
    engine = _isaac_engine()
    assert engine.robot_joint_names(name) == []
    assert engine.get_observation(robot_name=name) == {}


@pytest.mark.parametrize(("label", "name"), UNHASHABLE, ids=[lbl for lbl, _ in UNHASHABLE])
def test_isaac_raises_the_documented_miss_for_an_unhashable_camera(label, name):
    """``get_camera_params`` reports through an exception, so it must be the documented one.

    Its contract names ``KeyError`` for an unknown camera. A partial lookup
    replaced that with a ``TypeError`` out of the membership test itself, which
    no caller handling the documented failure would catch.
    """
    with pytest.raises(KeyError, match="not found"):
        _isaac_engine().get_camera_params(camera_name=name)


def test_isaac_resolves_a_registered_name_against_the_registry_alone():
    """A name that addresses an entity is answered from the registry, unchanged.

    These four need nothing but the registry, so a registered name gets the
    same answer it always did.
    """
    assert _isaac_engine().remove_robot("arm")["status"] == "success"
    assert _isaac_engine().remove_object("crate")["status"] == "success"
    assert _isaac_engine().remove_camera("look")["status"] == "success"
    assert _isaac_engine().robot_joint_names("arm") == ["pan"]
    assert (
        _isaac_engine().start_cameras_recording(output_dir="/tmp/isaac-name", cameras=["look"])["status"] == "success"
    )


@pytest.mark.parametrize(
    ("method", "call"),
    [
        ("send_action", lambda e: e.send_action({"pan": 0.1}, robot_name="arm")),
        ("install_action_controller", lambda e: e.install_action_controller("arm", object())),
        ("set_robot_pose", lambda e: e.set_robot_pose(robot_name="arm", position=[0.0, 0.0, 0.0])),
        ("set_joint_positions", lambda e: e.set_joint_positions({"pan": 0.1}, robot_name="arm")),
    ],
)
def test_isaac_carries_a_registered_name_past_the_lookup(method, call):
    """A registered name reaches the work, and fails on state, not on the name.

    Each of these goes on to read simulator state this skeleton does not build,
    so it still reports an error - but the error must no longer be the
    unknown-entity message. That distinction is what says the guard rejects only
    the names it should: a name the lookup turned away would have been reported
    as absent instead of getting this far.
    """
    result = call(_isaac_engine())
    assert result["status"] == "error"
    assert "not found" not in result["content"][0]["text"], (
        f"{method} reported a registered name as absent: {result['content'][0]['text']}"
    )


def test_isaac_reports_an_unknown_string_name_the_way_it_always_did():
    """A misspelled name keeps the message the method already had."""
    text = _isaac_engine().remove_object("crat")["content"][0]["text"]
    assert "Object 'crat' not found" in text


# --------------------------------------------------------------------------- #
# no backend may re-introduce a partial lookup                                #
# --------------------------------------------------------------------------- #

# Registries keyed by an entity name. A backend keeps them either on its
# :class:`SimWorld` (``objects`` / ``cameras`` / ``robots``) or on the engine
# itself, and the engine-level ones are just as reachable with a caller-supplied
# name: ``_policy_threads`` is consulted before the world registries by the guard
# that refuses a mutation while a policy runs, ``_action_controllers`` by
# ``send_action`` on the way to a task-space controller, and Isaac keeps its
# whole entity state in ``_robots`` / ``_objects`` / ``_cameras``. Listing only
# the world attributes would leave a backend that owns its registries outside
# the check while appearing to be inside it.
_REGISTRY_ATTRS = frozenset(
    {
        "objects",
        "cameras",
        "robots",
        "_robots",
        "_objects",
        "_cameras",
        "_policy_threads",
        "_action_controllers",
        "_cam_out_size",
    }
)

# Creating an entity is a different question from looking one up: there the name
# is not resolved but claimed, and a name that cannot be a key has to be refused
# rather than reported absent - which needs a contract for what a name may be,
# not just a total lookup. Those tests are left as they are, and pinned below so
# this exemption cannot outlive them.
_CREATION_FUNCTIONS = frozenset({"add_robot", "add_object", "add_camera"})


def _backend_dirs() -> list[Path]:
    """The backend packages, located from a symbol rather than a path literal."""
    simulation_dir = Path(inspect.getfile(registered)).parent
    return [simulation_dir / "mujoco", simulation_dir / "newton", simulation_dir / "isaac"]


def _is_registry(node: ast.AST) -> bool:
    """Whether ``node`` reads a name-keyed entity registry."""
    return isinstance(node, ast.Attribute) and node.attr in _REGISTRY_ATTRS


def _raw_lookups(source: str) -> list[tuple[str, int]]:
    """(enclosing function, line) for each registry lookup not using the helpers."""
    tree = ast.parse(source)
    owner: dict[int, str] = {}

    def annotate(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                annotate(child, child.name)
            else:
                if hasattr(child, "lineno"):
                    owner[child.lineno] = name
                annotate(child, name)

    annotate(tree, "<module>")
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(isinstance(op, ast.In | ast.NotIn) for op in node.ops):
            hit = any(_is_registry(c) for c in node.comparators)
        elif isinstance(node, ast.Call):
            fn = node.func
            hit = isinstance(fn, ast.Attribute) and fn.attr == "get" and _is_registry(fn.value)
        else:
            continue
        if hit:
            found.append((owner.get(node.lineno, "<module>"), node.lineno))
    return found


def test_every_backend_lookup_uses_the_shared_rule():
    """Only entity creation may test a registry directly."""
    offenders = []
    for directory in _backend_dirs():
        for path in sorted(directory.glob("*.py")):
            for function, line in _raw_lookups(path.read_text()):
                if function not in _CREATION_FUNCTIONS:
                    offenders.append(f"{path.name}:{line} in {function}()")
    assert not offenders, "these registry lookups are not total:\n" + "\n".join(f"  {o}" for o in offenders)


def test_the_creation_exemption_still_describes_real_code():
    """A stale exemption must be deleted, not left as a hole in the check."""
    exempt = {
        function
        for directory in _backend_dirs()
        for path in directory.glob("*.py")
        for function, _line in _raw_lookups(path.read_text())
    }
    assert exempt and exempt <= _CREATION_FUNCTIONS, f"the exemption no longer matches the code: found {sorted(exempt)}"


def test_the_check_detects_a_partial_lookup():
    """A scanner that silently matched nothing would look like a clean backend."""
    planted = "def look(self, name):\n    if name not in self._world.robots:\n        return None\n"
    assert _raw_lookups(planted) == [("look", 2)]
    guarded = "def look(self, name):\n    if not registered(self._world.robots, name):\n        return None\n"
    assert _raw_lookups(guarded) == []
