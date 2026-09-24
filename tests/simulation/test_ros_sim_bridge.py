"""Behavior tests for the ROS 2 simulation bridge (``SimEngine(ros2_bridge=...)``).

``rclpy`` is a system-provided, non-PyPI dependency, so these tests inject a
fake ``rclpy`` + ``sensor_msgs.msg`` into ``sys.modules`` to exercise the
publisher wiring with NO ROS 2 installed. They assert that:

* :class:`SimRosBridge` builds the right topics and ``JointState`` / ``Image``
  message fields and routes them to per-robot publishers.
* :meth:`SimEngine._publish_ros_telemetry` reads joint state from
  ``get_observation`` and forwards it, and is a no-op when the bridge is off.
* Enabling the bridge with no ``rclpy`` raises a clear :class:`ImportError`.
* :meth:`SimRosBridge.shutdown` releases the node handle *and* the rclpy context
  it initialized, so a failure destroying one does not leak the other.
* A bridge built in a process that already runs an ``rclpy`` context on another
  domain is refused, because a context's domain is fixed at ``init`` and the
  constructor's ``ROS_DOMAIN_ID`` write cannot move it. The guard lives in the
  shared :class:`~strands_robots.ros_telemetry.RosTelemetryBridge`, so the
  hardware bridge is held to it too - and, being the one refusal that lands
  after the constructor's ``ROS_DOMAIN_ID`` write, it hands that variable back
  as it found it.
* A ``MuJoCoSimEngine`` constructor argument that is refused leaves no rclpy
  node or context behind, whichever argument it is.
"""

from __future__ import annotations

import logging
import os
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import strands_robots.utils as utils_mod
from strands_robots.simulation.base import SimEngine


class _FakePublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[Any] = []

    def publish(self, msg: Any) -> None:
        self.messages.append(msg)


class _FakeClock:
    def now(self) -> Any:
        return self

    def to_msg(self) -> str:
        return "stamp"


class _FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.publishers: list[_FakePublisher] = []
        self.destroyed = False
        #: Set to model rclpy's ``InvalidHandle`` / ``RCLError`` out of the C layer.
        self.destroy_error: BaseException | None = None

    def get_clock(self) -> _FakeClock:
        return _FakeClock()

    def create_publisher(self, _msg_type: Any, topic: str, _depth: int) -> _FakePublisher:
        pub = _FakePublisher(topic)
        self.publishers.append(pub)
        return pub

    def destroy_node(self) -> None:
        if self.destroy_error is not None:
            raise self.destroy_error
        self.destroyed = True


class _Header:
    def __init__(self) -> None:
        self.stamp: Any = None
        self.frame_id: str = ""


class _JointState:
    def __init__(self) -> None:
        self.header = _Header()
        self.name: list[str] = []
        self.position: list[float] = []


class _Image:
    def __init__(self) -> None:
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.step = 0
        self.data = b""


class _FakeContext:
    """A context whose domain is fixed when it is initialized, as rclpy's is.

    ``ROS_DOMAIN_ID`` is read once, by ``rclpy.init``; a later write to the
    environment does not move a running context. Modelling that is what lets
    these tests grade the bridge's domain handling with no ROS 2 installed.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def get_domain_id(self) -> int:
        return int(self._state["domain"])


@pytest.fixture
def fake_ros(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject a fake rclpy + sensor_msgs.msg and clear require_optional's cache."""
    state: dict[str, Any] = {"inited": False, "shutdown": False, "nodes": [], "domain": 0}

    rclpy = ModuleType("rclpy")
    rclpy.ok = lambda: state["inited"]  # type: ignore[attr-defined]
    rclpy.get_default_context = lambda: _FakeContext(state)  # type: ignore[attr-defined]

    def _init() -> None:
        state["inited"] = True
        # The domain the context comes up on is the one in force at init.
        state["domain"] = int(os.environ.get("ROS_DOMAIN_ID", "0"))

    def _shutdown() -> None:
        state["shutdown"] = True
        state["inited"] = False

    def _create_node(name: str) -> _FakeNode:
        node = _FakeNode(name)
        state["nodes"].append(node)
        return node

    rclpy.init = _init  # type: ignore[attr-defined]
    rclpy.shutdown = _shutdown  # type: ignore[attr-defined]
    rclpy.create_node = _create_node  # type: ignore[attr-defined]

    sensor_pkg = ModuleType("sensor_msgs")
    sensor_msg = ModuleType("sensor_msgs.msg")
    sensor_msg.JointState = _JointState  # type: ignore[attr-defined]
    sensor_msg.Image = _Image  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_pkg)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msg)
    # require_optional memoizes resolved modules; drop any cached real entries.
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    return state


def test_sim_ros_bridge_publishes_joint_states(fake_ros: dict[str, Any]) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge(domain_id=7)
    bridge.publish_joint_states("so101", ["shoulder_pan", "elbow"], [0.1, 0.2])

    node = fake_ros["nodes"][0]
    (pub,) = node.publishers
    assert pub.topic == "/so101/joint_states"
    (msg,) = pub.messages
    assert msg.name == ["shoulder_pan", "elbow"]
    assert msg.position == [0.1, 0.2]
    assert msg.header.frame_id == "so101"


def test_sim_ros_bridge_sets_domain_env(fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    SimRosBridge(domain_id=42)
    assert os.environ["ROS_DOMAIN_ID"] == "42"


def test_sim_ros_bridge_publishes_rgb_image(fake_ros: dict[str, Any]) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge()
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    bridge.publish_image("so101", "front", frame)

    node = fake_ros["nodes"][0]
    (pub,) = node.publishers
    assert pub.topic == "/so101/front/image_raw"
    (msg,) = pub.messages
    assert (msg.height, msg.width, msg.encoding, msg.step) == (4, 5, "rgb8", 15)
    assert len(msg.data) == 4 * 5 * 3


def test_sim_ros_bridge_ignores_non_rgb_image(fake_ros: dict[str, Any]) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge()
    bridge.publish_image("so101", "depth", np.zeros((4, 5), dtype=np.uint8))
    assert fake_ros["nodes"][0].publishers == []


def test_sim_ros_bridge_shutdown_destroys_node(fake_ros: dict[str, Any]) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge()
    node = fake_ros["nodes"][0]
    bridge.shutdown()
    assert node.destroyed is True
    assert fake_ros["shutdown"] is True
    bridge.shutdown()  # idempotent


def test_shutdown_releases_the_context_when_the_node_will_not_die(fake_ros: dict[str, Any]) -> None:
    """A node that cannot be destroyed does not keep the rclpy context alive.

    ``shutdown`` releases two independent resources: this bridge's node handle
    and - when it was this bridge that called ``rclpy.init()`` - the
    process-wide context. Letting a ``destroy_node`` failure propagate skipped
    the second release, and nothing retried it: the bridge keeps claiming
    ownership while the next bridge in the process sees ``rclpy.ok()`` already
    true, disclaims ownership, and never shuts it down either. Both call sites
    (``SimEngine.cleanup`` and ``Robot.cleanup``) tear the bridge down inside a
    suppressing block, so the leaked participant was reported to no one.
    """
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge(domain_id=7)
    assert fake_ros["inited"] is True  # this bridge owns the context
    fake_ros["nodes"][0].destroy_error = RuntimeError("failed to destroy node: handle invalid")

    bridge.shutdown()  # best-effort: the node failure is not the caller's to handle

    assert fake_ros["shutdown"] is True, "the context release was skipped by the node failure"
    assert fake_ros["inited"] is False, "the rclpy context outlived the bridge that initialized it"
    # ...so ownership passes cleanly to the next bridge in this process.
    SimRosBridge(domain_id=7).shutdown()
    assert fake_ros["inited"] is False


#: ``(domain of the context already running, domain the caller asks for)``, and
#: whether the bridge can honor the request. A context reads ``ROS_DOMAIN_ID``
#: once, at ``init``, so only the domain already in force is reachable: the two
#: matching rows are the ones that must still build a bridge, so the guard cannot
#: be satisfied by refusing more than it should.
_DOMAIN_REQUESTS: list[tuple[int, int, bool]] = [
    (0, 7, False),  # the default domain in force, isolation asked for
    (7, 0, False),  # the reverse: a domain in force, the default asked for
    (7, 7, True),
    (0, 0, True),
]


@pytest.mark.parametrize(("running", "requested", "honored"), _DOMAIN_REQUESTS, ids=repr)
def test_a_domain_the_running_context_cannot_be_moved_to_is_refused(
    fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch, running: int, requested: int, honored: bool
) -> None:
    """The bridge publishes where the caller asked, or says why it cannot.

    The constructor pins its domain by writing ``ROS_DOMAIN_ID``, which only
    selects the domain of a context it starts itself. In a process that already
    has one - a ``rclpy`` node embedding the simulation, or the hardware bridge
    of the arm this sim mirrors - the bridge joined that context and the write
    was inert: measured on ROS 2 Jazzy, a bridge asked for domain 7 beside a
    context on domain 0 reported success with ``ROS_DOMAIN_ID=7`` and published
    every ``JointState`` on domain 0, where a subscriber on 7 never appeared. Two
    bridges for one robot then put two publishers on ``/<robot>/joint_states``
    and interleaved two poses under one name.

    """
    from strands_robots.simulation.ros_bridge import SimRosBridge

    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    first = SimRosBridge(domain_id=running)
    assert fake_ros["domain"] == running

    if honored:
        second = SimRosBridge(domain_id=requested)
        assert second._owns_context is False, "the running context is not this bridge's to release"
        assert len(fake_ros["nodes"]) == 2
        second.shutdown()
    else:
        with pytest.raises(ValueError) as excinfo:
            SimRosBridge(domain_id=requested)

        message = str(excinfo.value)
        assert f"domain_id {requested} cannot be honored" in message
        assert f"already has a running rclpy context on domain {running}" in message
        assert f"pass domain_id={running}" in message, "the refusal names the domain that would work"
        assert len(fake_ros["nodes"]) == 1, "the refused bridge created a node it cannot shut down"

    first.shutdown()


#: What ``ROS_DOMAIN_ID`` held before a bridge that gets refused was built. This
#: refusal is the only one that lands *after* the constructor's process-wide
#: write, so it is the only one that has to undo it, and "as it found it" has two
#: shapes: unset stays unset, and a value the shell or an earlier bridge left
#: comes back unchanged rather than being replaced by the domain in force.
_PREEXISTING_DOMAIN_ENV: list[str | None] = [None, "3"]


@pytest.mark.parametrize("preexisting", _PREEXISTING_DOMAIN_ENV, ids=repr)
def test_a_refused_domain_leaves_the_environment_as_it_found_it(
    fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch, preexisting: str | None
) -> None:
    """A refused bridge hands the caller's ``ROS_DOMAIN_ID`` back untouched.

    The range and QoS refusals land before the write and so leave the
    environment alone for free; this one cannot, and a stranded ``ROS_DOMAIN_ID``
    outlives the constructor that failed - every later context in the process,
    and every ROS 2 command line run from it, reads the domain of a bridge that
    was never built.

    The context here is one no bridge owns, as any library calling
    ``rclpy.init()`` leaves behind, so the variable under test is the caller's
    own rather than an earlier bridge's.
    """
    from strands_robots.simulation.ros_bridge import SimRosBridge

    if preexisting is None:
        monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    else:
        monkeypatch.setenv("ROS_DOMAIN_ID", preexisting)
    fake_ros["inited"] = True
    fake_ros["domain"] = 0

    with pytest.raises(ValueError, match="domain_id 7 cannot be honored"):
        SimRosBridge(domain_id=7)

    assert os.environ.get("ROS_DOMAIN_ID") == preexisting
    assert fake_ros["nodes"] == [], "the refused bridge created a node it cannot shut down"


def test_shutdown_reports_a_context_it_could_not_release(
    fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A context that will not shut down is reported, because nothing retries it.

    Unlike the node handle - which the context shutdown would have taken down
    anyway - a failed ``rclpy.shutdown`` is the last word on a participant still
    on the wire, so it is logged at warning rather than swallowed at debug.
    """
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge(domain_id=7)

    def _boom() -> None:
        raise RuntimeError("rcl_shutdown failed")

    monkeypatch.setattr(sys.modules["rclpy"], "shutdown", _boom)

    with caplog.at_level(logging.WARNING, logger="strands_robots.ros_telemetry"):
        bridge.shutdown()  # must not raise out of teardown

    assert "could not be shut down" in caplog.text
    assert "domain 7" in caplog.text


class _FakeEngine(SimEngine):
    """Minimal concrete engine exercising the telemetry helper only."""

    def __init__(
        self,
        observation: dict[str, Any],
        *,
        ros2_bridge: bool = False,
        ros2_domain: int = 0,
        joint_names: list[str] | None = None,
    ) -> None:
        self._obs = observation
        self._joint_names = ["shoulder_pan", "elbow"] if joint_names is None else joint_names
        self._init_ros_bridge(ros2_bridge=ros2_bridge, ros2_domain=ros2_domain)

    def list_robots(self) -> list[str]:
        return ["so101"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._joint_names)

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        return self._obs

    # Unused abstract methods for this focused test (never called).
    def create_world(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def destroy(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def add_robot(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_robot(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def add_object(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_object(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def send_action(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def physics_timestep(self) -> float:
        return 0.002


def test_publish_telemetry_forwards_joint_state(fake_ros: dict[str, Any]) -> None:
    obs = {"shoulder_pan": 0.5, "elbow": -0.25, "front": np.zeros((2, 2, 3), dtype=np.uint8)}
    engine = _FakeEngine(obs, ros2_bridge=True, ros2_domain=3)
    engine._publish_ros_telemetry()

    node = fake_ros["nodes"][0]
    topics = {p.topic: p for p in node.publishers}
    assert "/so101/joint_states" in topics
    js = topics["/so101/joint_states"].messages[0]
    assert js.position == [0.5, -0.25]
    assert "/so101/front/image_raw" in topics


class TestJointStateArraysNameTheSameJoints:
    """A published ``JointState``'s two arrays must describe the same joints.

    ``sensor_msgs/JointState`` pairs ``name`` and ``position`` by index, so the
    telemetry helper cannot filter one column and not the other. Every
    floating-base robot in the registry reaches this: the root freejoint is
    element 0 of ``robot_joint_names()`` and is not an observation key, so a
    positions-only filter compacted the value column and published every later
    joint under the name of the one before it.
    """

    #: A floating-base robot's joint list: the root joint carries no observation.
    FLOATING_BASE = ["freejoint", "hip", "knee", "ankle"]

    @staticmethod
    def _published(fake_ros: dict[str, Any]) -> Any:
        node = fake_ros["nodes"][0]
        topics = {p.topic: p for p in node.publishers}
        assert "/so101/joint_states" in topics, sorted(topics)
        return topics["/so101/joint_states"].messages[0]

    def test_a_joint_absent_from_the_observation_drops_its_name_too(self, fake_ros: dict[str, Any]) -> None:
        obs = {"hip": 0.1, "knee": 0.2, "ankle": 0.3}
        engine = _FakeEngine(obs, ros2_bridge=True, joint_names=self.FLOATING_BASE)
        engine._publish_ros_telemetry()

        js = self._published(fake_ros)
        # Premise: the fixture really does exercise a gap.
        assert "freejoint" not in obs
        assert len(js.name) == len(js.position), (js.name, js.position)
        # Every reported joint carries ITS OWN value, not its neighbour's.
        assert dict(zip(js.name, js.position, strict=True)) == obs
        assert "freejoint" not in js.name

    def test_no_joint_is_reported_under_its_neighbours_name(self, fake_ros: dict[str, Any]) -> None:
        # Distinct values so a one-place shift cannot pass by coincidence.
        obs = {"hip": -0.75, "knee": 0.5, "ankle": 1.25}
        engine = _FakeEngine(obs, ros2_bridge=True, joint_names=self.FLOATING_BASE)
        engine._publish_ros_telemetry()

        wire = dict(zip(self._published(fake_ros).name, self._published(fake_ros).position, strict=True))
        for joint, truth in obs.items():
            assert wire[joint] == truth, f"{joint} published as {wire[joint]}, truth {truth}"
        # The tail joint is reported at all, rather than falling off the end.
        assert "ankle" in wire

    def test_a_fully_observed_robot_reports_every_joint(self, fake_ros: dict[str, Any]) -> None:
        # Control: no gap, so nothing is dropped and the name column is full.
        obs = {"shoulder_pan": 0.5, "elbow": -0.25}
        engine = _FakeEngine(obs, ros2_bridge=True)
        engine._publish_ros_telemetry()

        js = self._published(fake_ros)
        assert js.name == ["shoulder_pan", "elbow"]
        assert js.position == [0.5, -0.25]


def test_publish_telemetry_is_noop_when_disabled(fake_ros: dict[str, Any]) -> None:
    engine = _FakeEngine({"shoulder_pan": 0.0, "elbow": 0.0})  # ros2_bridge defaults False
    engine._publish_ros_telemetry()
    assert engine._ros_bridge is None
    assert fake_ros["nodes"] == []


def test_enabling_bridge_without_rclpy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # No fake_ros fixture here: ensure rclpy is absent and the cache is clear.
    monkeypatch.setitem(sys.modules, "rclpy", None)
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    with pytest.raises(ImportError):
        _FakeEngine({}, ros2_bridge=True)


def test_publish_telemetry_safe_without_bridge_init(fake_ros: dict[str, Any]) -> None:
    """Telemetry hooks tolerate an engine that never called ``_init_ros_bridge``.

    The simulation ABC defines no ``__init__``, so a lightweight subclass (or a
    backend constructed via an unusual path) may never initialize the bridge
    attributes. ``step`` calls ``_publish_ros_telemetry`` unconditionally, so it
    - and ``_shutdown_ros_bridge`` - must be safe no-ops in that case rather
    than raising ``AttributeError`` on a missing ``_ros_bridge``. Here we strip
    the attribute off a normally-built engine to model that uninitialized state.
    """
    engine = _FakeEngine({"shoulder_pan": 0.0, "elbow": 0.0})
    del engine._ros_bridge

    engine._publish_ros_telemetry()  # must not raise
    engine._shutdown_ros_bridge()  # must not raise
    assert fake_ros["nodes"] == []


class _TwoRobotEngine(SimEngine):
    """Engine with two robots where one robot's observation always fails.

    Models a transient per-robot render fault (EGL/GL context loss, a camera
    that produced no frame) so we can assert the documented contract: a single
    robot's failure must not interrupt the loop or escape ``step()``.
    """

    def __init__(self, *, ros2_bridge: bool = True, ros2_domain: int = 0) -> None:
        self._init_ros_bridge(ros2_bridge=ros2_bridge, ros2_domain=ros2_domain)

    def list_robots(self) -> list[str]:
        return ["broken", "healthy"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["shoulder_pan", "elbow"]

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        if robot_name == "broken":
            raise RuntimeError("camera render failed (EGL context lost)")
        return {"shoulder_pan": 0.5, "elbow": -0.25}

    # Unused abstract methods for this focused test (never called).
    def create_world(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def destroy(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def add_robot(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_robot(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def add_object(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_object(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def send_action(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def physics_timestep(self) -> float:
        return 0.002


def test_publish_telemetry_per_robot_failure_does_not_interrupt_loop(fake_ros: dict[str, Any]) -> None:
    """A failing robot is skipped; healthy robots still publish; step() never crashes.

    Pins the docstring contract on the hot ``ros2_bridge=True`` path: per-robot
    failures (e.g. a camera that did not render) never interrupt the loop. The
    "broken" robot raises in ``get_observation``; the "healthy" robot - listed
    after it - must still publish its joint_states, and no exception escapes.
    """
    engine = _TwoRobotEngine(ros2_bridge=True, ros2_domain=5)

    engine._publish_ros_telemetry()  # must not raise despite the broken robot

    node = fake_ros["nodes"][0]
    topics = {p.topic: p for p in node.publishers}
    # Broken robot never published; healthy one published despite being second.
    assert "/broken/joint_states" not in topics
    assert "/healthy/joint_states" in topics
    assert topics["/healthy/joint_states"].messages[0].position == [0.5, -0.25]


def test_publish_telemetry_skip_images_publishes_joints_only(fake_ros: dict[str, Any]) -> None:
    """``skip_images=True`` publishes joint_states but never publishes camera images.

    Backends call ``_publish_ros_telemetry(skip_images=True)`` on the hot step
    path to skip the render/publish cost when no camera subscriber needs frames.
    Pin the contract: joint_states still go out, but no ``image_raw`` publisher
    is created for camera keys present in the observation.
    """
    obs = {"shoulder_pan": 0.5, "elbow": -0.25, "front": np.zeros((2, 2, 3), dtype=np.uint8)}
    engine = _FakeEngine(obs, ros2_bridge=True, ros2_domain=3)

    engine._publish_ros_telemetry(skip_images=True)

    node = fake_ros["nodes"][0]
    topics = {p.topic: p for p in node.publishers}
    assert "/so101/joint_states" in topics
    assert topics["/so101/joint_states"].messages[0].position == [0.5, -0.25]
    assert "/so101/front/image_raw" not in topics


def test_shutdown_ros_bridge_tears_down_active_bridge_idempotently(fake_ros: dict[str, Any]) -> None:
    """``_shutdown_ros_bridge`` destroys an active bridge and clears the handle.

    Pins the documented "safe to call repeatedly" contract: the first call
    forwards to :meth:`SimRosBridge.shutdown` (destroying the ROS 2 node) and
    resets ``_ros_bridge`` to None, so a second call is a no-op rather than a
    double shutdown on a dead node.
    """
    engine = _FakeEngine({"shoulder_pan": 0.0, "elbow": 0.0}, ros2_bridge=True, ros2_domain=2)
    node = fake_ros["nodes"][0]
    assert engine._ros_bridge is not None

    engine._shutdown_ros_bridge()

    assert node.destroyed is True
    assert engine._ros_bridge is None

    engine._shutdown_ros_bridge()  # idempotent: must not raise, handle stays None
    assert engine._ros_bridge is None


# -- a refused constructor argument must not leave the bridge behind ---------
#
# ``MuJoCoSimEngine.__init__`` builds the optional bridge part-way through, so
# every argument it can refuse has to be answered above that point. These pin
# the consequence rather than the ordering: ``__init__`` raising returns no
# object, so the ``cleanup()`` that would call ``_shutdown_ros_bridge`` is
# unreachable, and anything the refused call built is leaked for the life of the
# process.

REFUSED_ARGUMENTS = [
    ({"default_width": 0}, ValueError, "default_width"),
    ({"default_height": -1}, ValueError, "default_height"),
    ({"ros2_domain": 233}, ValueError, "ros2_domain"),
    ({"mesh": True}, TypeError, "mesh="),
]


@pytest.mark.parametrize(("kwargs", "exc", "match"), REFUSED_ARGUMENTS)
def test_a_refused_constructor_argument_leaves_no_ros_node_behind(
    kwargs: dict[str, Any], exc: type[BaseException], match: str, fake_ros: dict[str, Any]
) -> None:
    """Every refusable ``MuJoCoSimEngine`` argument is answered before the bridge.

    ``mesh=True`` is the documented "not a mesh client" refusal, and it was
    raised *after* ``_init_ros_bridge`` had created the ``strands_sim`` node and
    initialized the rclpy context. The other three rows are the arguments whose
    guards already precede the bridge, so they hold either way and say that this
    is one placement rule rather than one special case.
    """
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    with pytest.raises(exc, match=match):
        MuJoCoSimEngine(ros2_bridge=True, **kwargs)

    assert fake_ros["nodes"] == []
    assert fake_ros["inited"] is False


def test_a_retry_after_a_refusal_still_releases_the_context_it_initialized(
    fake_ros: dict[str, Any],
) -> None:
    """A corrected second call leaves the process as it found it.

    This is why the guard has to precede the bridge rather than clean up after
    it. A bridge records ``_owns_context`` only when it is the one that called
    ``rclpy.init``, so a bridge leaked by a refused construction keeps that
    ownership: the corrected retry finds the context already up, declines it,
    and on teardown destroys only its own node - leaving the leaked node alive
    and the context initialized with nobody left to shut it down.
    """
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    with pytest.raises(TypeError, match="mesh="):
        MuJoCoSimEngine(ros2_bridge=True, mesh=True)

    engine = MuJoCoSimEngine(ros2_bridge=True)
    engine.cleanup()

    assert [node.destroyed for node in fake_ros["nodes"]] == [True]
    assert fake_ros["shutdown"] is True
    assert fake_ros["inited"] is False


def test_a_mesh_client_the_engine_can_stop_is_still_accepted(fake_ros: dict[str, Any]) -> None:
    """The guard moved, it did not tighten: a stoppable client is stored and stopped.

    Anti-vacuity companion to the rows above - a guard that refused everything
    would satisfy them all - and it pins that resolving the handle earlier does
    not change where it lands or who stops it.
    """
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    stopped: list[bool] = []
    client = type("_MeshClient", (), {"stop": lambda _self: stopped.append(True)})()

    engine = MuJoCoSimEngine(ros2_bridge=True, mesh=client)
    assert engine.mesh is client

    engine.cleanup()

    assert stopped == [True]
    assert [node.destroyed for node in fake_ros["nodes"]] == [True]
