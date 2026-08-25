"""One faulting sensor must not erase the capabilities surveyed after it.

``Mesh._build_presence`` advertises a ``topics`` list naming the extended
telemetry a peer can serve -- ``pose``, ``imu``, ``odom``, ``lidar``,
``health``, ``hand``, ``map``. Each entry is decided by probing a provider
attribute on the robot (``_pose``, ``_imu``, ...), and on hardware those are
properties that read a live sensor bus rather than plain fields: one can raise
while every other sensor stays readable. A non-``AttributeError`` propagates
through ``getattr(robot, name, None)``, so such a fault reaches the survey.

The capability list is the only place these topics are announced, so a peer
that under-reports is a peer whose readable telemetry is never subscribed to.
Surveying every provider under one shared ``try`` made the first fault
abandon the rest of the survey: a robot whose ``_pose`` faulted advertised
none of the IMU, lidar, hand or map data it was still serving. These cells
grade the survey at the granularity the failure has -- one provider.

The pre-existing pins covered only the two extremes, where a shared guard and
per-attribute guards are indistinguishable: ``tests/mesh/test_mesh.py``
drives a robot whose every attribute raises (``topics == ["health"]``) and
``tests/mesh/test_deep_mesh.py`` drives one whose every sensor answers (all
seven topics). The mixed robot -- one fault, the rest healthy -- is the case
that separates them, and is what this file adds.

The topic-to-provider map is derived from ``_build_presence`` itself rather
than restated, so a topic that later gains or loses a provider is graded on
arrival instead of silently escaping these cells.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest

from strands_robots.mesh.core import Mesh, _sensor_present

# ── The surveyed capabilities, derived from the code under test ──────────────


def _topic_providers() -> dict[str, tuple[str, ...]]:
    """Map each advertised topic to the provider attributes that back it.

    Read out of :meth:`Mesh._build_presence` by locating every
    ``available_topics.append("<topic>")`` and collecting the attribute-name
    string constants in the ``if`` test that guards it.

    The collection is deliberately blind to *how* the guard reads those
    attributes -- a name mentioned in a ``_sensor_present(...)`` call and one
    mentioned in a bare ``getattr(...)`` are both found. So this map describes
    the same survey whichever way it is written, and the cells below grade the
    survey's behaviour rather than its spelling.
    :class:`TestEveryProviderIsProbedUnderItsOwnGuard` is what holds the
    spelling to the tolerant probe.

    Returns:
        Topic name to the provider attribute names probed for it, in the order
        the survey probes them.
    """
    body = textwrap.dedent(inspect.getsource(Mesh._build_presence))
    out: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(ast.parse(body)):
        if not isinstance(node, ast.If):
            continue
        appended = [
            stmt.value.args[0].value
            for stmt in node.body
            if isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "append"
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "available_topics"
            and stmt.value.args
            and isinstance(stmt.value.args[0], ast.Constant)
        ]
        if len(appended) != 1:
            continue
        topic = appended[0]
        assert isinstance(topic, str)
        attrs = tuple(
            n.value
            for n in ast.walk(node.test)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("_")
        )
        if attrs:
            out[topic] = attrs
    return out


TOPIC_PROVIDERS = _topic_providers()
ALL_PROVIDERS = tuple(attr for attrs in TOPIC_PROVIDERS.values() for attr in attrs)

# A value each provider can plausibly answer with. Only "not None" matters to
# the survey, but a shape the readers would accept keeps the double honest.
_READINGS: dict[str, Any] = {
    "_pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "_slam_pose": {"x": 1.0, "y": 2.0, "theta": 0.0},
    "_odom_pose": {"x": 0.5, "y": 0.0, "theta": 0.1},
    "_imu": {"accel": [0.0, 0.0, 9.81], "gyro": [0.0, 0.0, 0.0]},
    "_odom": {"vx": 0.0, "vy": 0.0},
    "_lidar_summary": {"min_range": 0.4},
    "_lidar_state": {"ranges": [1.0, 1.1]},
    "_battery": {"pct": 87, "charging": False},
    "_hands": {"left": {"grip": 0.0}},
    "_map_info": {"width": 10, "height": 10},
}


class _SensorRobot:
    """A robot whose providers answer, are absent, or fault -- independently.

    Every provider is a property, which is what a hardware robot exposes: the
    read reaches a sensor bus at probe time and so can fail at probe time.
    """

    def __init__(self, *, faulting: frozenset[str] = frozenset(), present: tuple[str, ...] = ALL_PROVIDERS) -> None:
        self._faulting = faulting
        self._present = present

    def _probe(self, name: str) -> Any:
        if name in self._faulting:
            raise RuntimeError(f"sensor bus fault reading {name!r}")
        if name not in self._present:
            return None
        return _READINGS[name]

    @property
    def _pose(self) -> Any:
        return self._probe("_pose")

    @property
    def _slam_pose(self) -> Any:
        return self._probe("_slam_pose")

    @property
    def _odom_pose(self) -> Any:
        return self._probe("_odom_pose")

    @property
    def _imu(self) -> Any:
        return self._probe("_imu")

    @property
    def _odom(self) -> Any:
        return self._probe("_odom")

    @property
    def _lidar_summary(self) -> Any:
        return self._probe("_lidar_summary")

    @property
    def _lidar_state(self) -> Any:
        return self._probe("_lidar_state")

    @property
    def _battery(self) -> Any:
        return self._probe("_battery")

    @property
    def _hands(self) -> Any:
        return self._probe("_hands")

    @property
    def _map_info(self) -> Any:
        return self._probe("_map_info")


def _topics(robot: Any) -> set[str]:
    """Return the advertised capability set for *robot*."""
    return set(Mesh(robot, peer_id="probe-peer", peer_type="robot")._build_presence().get("topics", []))


def _expected_topics(faulting: frozenset[str]) -> set[str]:
    """The capability set a robot with *faulting* providers can still serve.

    A topic survives while any one of its providers still answers; ``health``
    is advertised unconditionally because the payload carries host stats even
    when the robot itself reports no battery.
    """
    survivors = {topic for topic, attrs in TOPIC_PROVIDERS.items() if any(a not in faulting for a in attrs)}
    return survivors | {"health"}


class TestThePremisesTheSurveyRestsOn:
    """Without these the cells below would pass for the wrong reason."""

    def test_a_faulting_provider_reaches_the_survey(self) -> None:
        """A non-``AttributeError`` is not absorbed by ``getattr``'s default.

        This is why the survey needs a guard at all: a sensor-bus fault is not
        a missing attribute, so the three-argument ``getattr`` re-raises it.
        """
        robot = _SensorRobot(faulting=frozenset({"_pose"}))

        with pytest.raises(RuntimeError, match="sensor bus fault reading '_pose'"):
            getattr(robot, "_pose", None)

    def test_the_double_answers_every_derived_provider(self) -> None:
        """Every provider the survey probes has a reading and a live property."""
        assert set(ALL_PROVIDERS) <= set(_READINGS)
        healthy = _SensorRobot()
        for attr in ALL_PROVIDERS:
            assert getattr(healthy, attr) is not None, attr

    def test_at_least_one_topic_has_several_providers(self) -> None:
        """``pose`` is backed by three, so the any-provider case is not vacuous."""
        assert max(len(a) for a in TOPIC_PROVIDERS.values()) > 1
        assert len(TOPIC_PROVIDERS["pose"]) == 3

    def test_a_fully_readable_robot_advertises_every_topic(self) -> None:
        """Non-vacuity: the survey really does find all seven capabilities."""
        assert _topics(_SensorRobot()) == set(TOPIC_PROVIDERS) | {"health"}


class TestOneFaultingProviderKeepsTheOtherTopics:
    """The regression: the survey degrades per provider, not all at once."""

    @pytest.mark.parametrize("faulted", ALL_PROVIDERS)
    def test_the_remaining_capabilities_are_still_advertised(self, faulted: str) -> None:
        """Faulting one provider costs only the topics it alone backs.

        Every other capability is readable and must still be announced -- the
        capability list is the only place a peer declares them.
        """
        faulting = frozenset({faulted})

        assert _topics(_SensorRobot(faulting=faulting)) == _expected_topics(faulting)

    @pytest.mark.parametrize("faulted", ALL_PROVIDERS)
    def test_the_survey_still_completes(self, faulted: str) -> None:
        """The presence payload is built at all, and keeps its constant fields."""
        mesh = Mesh(_SensorRobot(faulting=frozenset({faulted})), peer_id="probe-peer", peer_type="robot")

        payload = mesh._build_presence()

        assert payload["robot_id"] == "probe-peer"
        assert payload["robot_type"] == "robot"

    def test_a_faulting_head_pose_still_advertises_the_imu(self) -> None:
        """The reported case, stated directly rather than only as a parameter.

        An expressive head whose pose provider faults still serves its IMU, and
        ``pose`` is surveyed first -- so under a shared guard this robot
        advertised ``health`` alone.
        """
        head = _SensorRobot(
            faulting=frozenset({"_pose", "_slam_pose", "_odom_pose"}),
            present=("_pose", "_slam_pose", "_odom_pose", "_imu", "_battery"),
        )

        topics = _topics(head)

        assert "imu" in topics
        assert "health" in topics
        assert "pose" not in topics


class TestATopicWithSeveralProvidersSurvivesOneFault:
    """``pose`` is served by three providers; one fault must not retire it."""

    def test_a_sibling_provider_still_answers_for_the_topic(self) -> None:
        robot = _SensorRobot(faulting=frozenset({"_pose"}))

        assert "pose" in _topics(robot)

    def test_the_topic_retires_only_when_every_provider_is_gone(self) -> None:
        robot = _SensorRobot(faulting=frozenset(TOPIC_PROVIDERS["pose"]))

        assert "pose" not in _topics(robot)


class TestTheAdvertisementDoesNotOverReach:
    """Tolerating a fault must not invent a capability the peer cannot serve."""

    def test_a_robot_with_no_sensors_advertises_health_only(self) -> None:
        """``health`` is the unconditional entry; nothing else is claimed."""
        assert _topics(_SensorRobot(present=())) == {"health"}

    def test_a_robot_whose_every_provider_faults_advertises_health_only(self) -> None:
        """Pins the behaviour ``tests/mesh/test_mesh.py`` already relies on."""
        assert _topics(_SensorRobot(faulting=frozenset(ALL_PROVIDERS))) == {"health"}

    def test_a_provider_answering_none_is_not_a_capability(self) -> None:
        """``None`` means "no reading", which is not the same as a fault."""
        robot = _SensorRobot(present=tuple(a for a in ALL_PROVIDERS if a != "_imu"))

        assert "imu" not in _topics(robot)

    def test_health_is_advertised_even_when_the_battery_faults(self) -> None:
        """Host stats are served regardless, so the topic is always available."""
        assert "health" in _topics(_SensorRobot(faulting=frozenset({"_battery"})))


class TestEveryProviderIsProbedUnderItsOwnGuard:
    """Structural: the granularity cannot regress to one shared ``try``."""

    @staticmethod
    def _survey() -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(inspect.getsource(Mesh._build_presence)))
        node = tree.body[0]
        assert isinstance(node, ast.FunctionDef)
        return node

    def test_no_provider_is_read_by_a_bare_getattr(self) -> None:
        """Every provider read goes through the tolerant probe.

        A provider read directly would carry its fault into the survey, which
        is the defect this file exists for.
        """
        bare = [
            ast.unparse(node)
            for node in ast.walk(self._survey())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in ALL_PROVIDERS
        ]

        assert bare == []

    def test_the_capability_survey_is_not_wrapped_in_a_shared_guard(self) -> None:
        """A ``try`` around the survey is what made one fault end it."""
        survey = self._survey()
        guarded_spans = [
            (node.lineno, node.end_lineno or node.lineno)
            for node in ast.walk(survey)
            if isinstance(node, ast.Try) and node.body
        ]
        probes = [
            node.lineno
            for node in ast.walk(survey)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_sensor_present"
        ]
        assert probes, "the survey must probe through _sensor_present"

        inside = [line for line in probes if any(lo <= line <= hi for lo, hi in guarded_spans)]

        assert inside == []


class TestTheProbeItself:
    """``_sensor_present`` reports readability, one attribute at a time."""

    def test_an_answering_attribute_reports_present(self) -> None:
        assert _sensor_present(_SensorRobot(), "_imu") is True

    def test_an_absent_attribute_reports_absent(self) -> None:
        assert _sensor_present(_SensorRobot(), "_nothing_here") is False

    def test_a_none_reading_reports_absent(self) -> None:
        assert _sensor_present(_SensorRobot(present=()), "_imu") is False

    def test_a_faulting_attribute_reports_absent_rather_than_raising(self) -> None:
        assert _sensor_present(_SensorRobot(faulting=frozenset({"_imu"})), "_imu") is False

    def test_a_faulting_attribute_does_not_stop_a_later_one(self) -> None:
        """The any-provider search continues past a fault to its siblings."""
        robot = _SensorRobot(faulting=frozenset({"_pose"}))

        assert _sensor_present(robot, "_pose", "_slam_pose") is True

    def test_the_fault_is_logged_for_diagnosis(self, caplog: pytest.LogCaptureFixture) -> None:
        """A dropped provider is debuggable rather than wholly invisible."""
        with caplog.at_level("DEBUG", logger="strands_robots.mesh.core"):
            _sensor_present(_SensorRobot(faulting=frozenset({"_imu"})), "_imu")

        assert any("_imu" in record.getMessage() for record in caplog.records)
