"""Behavior tests for the pure-RTPS hardware bridge (``HardwareRtpsBridge``).

``cyclonedds`` is an optional pip dependency, so these tests inject a fake
``cyclonedds`` (domain/pub/sub/topic) into ``sys.modules`` to exercise the
bridge wiring with NO cyclonedds installed. They assert that the RTPS bridge:

* publishes ``/<robot>/joint_states`` (+ per-camera ``image_raw``) with the
  ROS-mangled DDS topic names and the right ``sensor_msgs`` field layout;
* subscribes ``/<robot>/joint_command`` and forwards an inbound JointState into
  ``robot.send_action`` as a flat ``{name: pos}`` dict, ignoring empty/mismatched
  samples;
* is telemetry-only when no robot is bound or ``enable_commands=False``;
* tears down cleanly and is idempotent;
* raises a clear ImportError when cyclonedds is absent.

The IDL bundle's own type layouts are validated separately against a real ROS 2
node (see the PR's live cross-stack verification); here the IDL classes are
trivial fakes so the tests stay ROS-free.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

import strands_robots.utils as utils_mod


class _FakeWriter:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.samples: list[Any] = []

    def write(self, msg: Any) -> None:
        self.samples.append(msg)


class _FakeReader:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._queue: list[Any] = []

    def feed(self, msg: Any) -> None:
        self._queue.append(msg)

    def take(self, N: int = 10) -> list[Any]:
        out, self._queue = self._queue[:N], self._queue[N:]
        return out


class _FakeTopic:
    def __init__(self, participant: Any, name: str, idl_cls: Any) -> None:
        self.name = name
        self.idl_cls = idl_cls


class _FakeParticipant:
    def __init__(self, domain_id: int = 0, qos: Any = None) -> None:
        self.domain_id = domain_id
        self.qos = qos


# Minimal IDL stand-ins (the real layouts are validated against live ROS 2).
class _JointState:
    def __init__(self, header=None, name=None, position=None, velocity=None, effort=None) -> None:
        self.header = header
        self.name = name or []
        self.position = position or []
        self.velocity = velocity or []
        self.effort = effort or []


class _Image:
    def __init__(self, header=None, height=0, width=0, encoding="", is_bigendian=0, step=0, data=b"") -> None:
        self.header = header
        self.height = height
        self.width = width
        self.encoding = encoding
        self.is_bigendian = is_bigendian
        self.step = step
        self.data = data


class _Header:
    def __init__(self, stamp=None, frame_id="") -> None:
        self.stamp = stamp
        self.frame_id = frame_id


class _Time:
    def __init__(self, sec=0, nanosec=0) -> None:
        self.sec = sec
        self.nanosec = nanosec


_IDL = {
    "sensor_msgs/msg/JointState": _JointState,
    "sensor_msgs/msg/Image": _Image,
    "std_msgs/msg/Header": _Header,
    "builtin_interfaces/msg/Time": _Time,
}


@pytest.fixture
def fake_cyclonedds(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject a fake cyclonedds + patch the IDL bundle's get_type/have_cyclonedds."""
    state: dict[str, Any] = {"writers": [], "readers": [], "participants": []}

    # require_optional("cyclonedds") must succeed -> inject a module + clear cache.
    cyclonedds = ModuleType("cyclonedds")
    domain_mod = ModuleType("cyclonedds.domain")
    pub_mod = ModuleType("cyclonedds.pub")
    sub_mod = ModuleType("cyclonedds.sub")
    topic_mod = ModuleType("cyclonedds.topic")

    def _make_participant(domain_id: int = 0, qos: Any = None) -> _FakeParticipant:
        p = _FakeParticipant(domain_id, qos=qos)
        state["participants"].append(p)
        return p

    def _make_writer(_participant: Any, topic: _FakeTopic) -> _FakeWriter:
        w = _FakeWriter(topic.name)
        state["writers"].append(w)
        return w

    def _make_reader(_participant: Any, topic: _FakeTopic) -> _FakeReader:
        r = _FakeReader(topic.name)
        state["readers"].append(r)
        return r

    domain_mod.DomainParticipant = _make_participant  # type: ignore[attr-defined]
    pub_mod.DataWriter = _make_writer  # type: ignore[attr-defined]
    sub_mod.DataReader = _make_reader  # type: ignore[attr-defined]
    topic_mod.Topic = _FakeTopic  # type: ignore[attr-defined]

    # Fake cyclonedds.qos so _build_security_qos can assemble Property policies.
    qos_mod = ModuleType("cyclonedds.qos")

    class _FakeQos:
        def __init__(self, *policies: Any) -> None:
            self.policies = list(policies)

    class _FakePolicy:
        class Property:
            def __init__(self, name: str, value: str) -> None:
                self.name = name
                self.value = value

    qos_mod.Qos = _FakeQos  # type: ignore[attr-defined]
    qos_mod.Policy = _FakePolicy  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cyclonedds", cyclonedds)
    monkeypatch.setitem(sys.modules, "cyclonedds.domain", domain_mod)
    monkeypatch.setitem(sys.modules, "cyclonedds.pub", pub_mod)
    monkeypatch.setitem(sys.modules, "cyclonedds.sub", sub_mod)
    monkeypatch.setitem(sys.modules, "cyclonedds.topic", topic_mod)
    monkeypatch.setitem(sys.modules, "cyclonedds.qos", qos_mod)
    monkeypatch.setattr(utils_mod, "_lazy_modules", {"cyclonedds": cyclonedds}, raising=False)

    # Patch the IDL bundle resolver so it returns our trivial classes.
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda t: _IDL[t], raising=True)
    monkeypatch.setattr(idl_mod, "have_cyclonedds", lambda: True, raising=True)

    # The command-behavior tests below exercise the inbound surface, which is
    # now gated on DDS Security. They are not security tests, so run them under
    # the explicit insecure opt-out; the dedicated gate tests control this env
    # and a dds_security_config themselves.
    monkeypatch.setenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", "1")
    return state


class _FakeInner:
    name = "test_arm"


class _FakeRobot:
    def __init__(self) -> None:
        self.tool_name_str = "test_arm"
        self.robot = _FakeInner()
        self.sent_actions: list[dict[str, Any]] = []

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.sent_actions.append(action)
        return {"status": "success", "content": [{"text": "ok"}]}


def _bridge(robot=None, **kw):
    from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

    return HardwareRtpsBridge(robot, **kw)  # type: ignore[arg-type]


def test_publish_joint_states_uses_mangled_topic_and_fields(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge(enable_commands=False)
    b.publish_joint_states("test_arm", ["a", "b"], [0.1, 0.2])
    writer = next(w for w in fake_cyclonedds["writers"] if w.topic == "rt/test_arm/joint_states")
    (msg,) = writer.samples
    assert msg.name == ["a", "b"]
    assert msg.position == [0.1, 0.2]
    assert msg.header.frame_id == "test_arm"


class TestJointStateArraysAreRefusedWhenTheyDisagree:
    """A ``JointState`` whose two arrays name different joints never reaches the wire.

    ``name`` and ``position`` are paired by index, so a caller that supplies a
    different number of each has no pose to publish: a consumer's
    ``zip(name, position)`` reports every joint after the gap under its
    neighbour's name and the tail unreported. Dropped whole, with the reason
    logged, for the reason :meth:`_command_action` refuses a malformed inbound
    command whole rather than applying part of it.
    """

    @pytest.mark.parametrize(
        ("names", "positions"),
        [
            (["hip", "knee", "ankle"], [0.1, 0.2]),
            (["hip", "knee"], [0.1, 0.2, 0.3]),
            (["hip"], []),
        ],
    )
    def test_a_mismatched_pair_writes_nothing(
        self,
        fake_cyclonedds: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
        names: list[str],
        positions: list[float],
    ) -> None:
        b = _bridge(enable_commands=False)
        with caplog.at_level(logging.WARNING):
            b.publish_joint_states("test_arm", names, positions)

        assert [s for w in fake_cyclonedds["writers"] for s in w.samples] == []
        assert f"{len(names)} joint name(s) and {len(positions)} position(s)" in caplog.text
        assert "no partial publication" in caplog.text

    def test_a_refused_pair_advertises_no_writer(self, fake_cyclonedds: dict[str, Any]) -> None:
        # The refusal is before the lazy writer, so a topic nothing can be
        # published on is never advertised either.
        b = _bridge(enable_commands=False)
        b.publish_joint_states("test_arm", ["hip", "knee"], [0.1])
        assert [w.topic for w in fake_cyclonedds["writers"]] == []

    def test_a_matching_pair_still_writes(self, fake_cyclonedds: dict[str, Any]) -> None:
        # Control: the guard refuses no more than the mismatch.
        b = _bridge(enable_commands=False)
        b.publish_joint_states("test_arm", ["hip", "knee"], [0.1, 0.2])
        writer = next(w for w in fake_cyclonedds["writers"] if w.topic == "rt/test_arm/joint_states")
        (msg,) = writer.samples
        assert msg.name == ["hip", "knee"]
        assert msg.position == [0.1, 0.2]


def test_publish_image_fields(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge(enable_commands=False)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    b.publish_image("test_arm", "wrist", frame)
    writer = next(w for w in fake_cyclonedds["writers"] if w.topic == "rt/test_arm/wrist/image_raw")
    (msg,) = writer.samples
    assert (msg.height, msg.width, msg.encoding, msg.step) == (4, 6, "rgb8", 6 * 3)
    assert len(msg.data) == 4 * 6 * 3


def test_publish_image_drops_non_rgb_frames(fake_cyclonedds: dict[str, Any]) -> None:
    """publish_image silently drops frames that are not HxWx3 RGB.

    A ``sensor_msgs/Image`` published with a mismatched shape/step would make a
    downstream ROS 2 consumer misread the raw byte buffer, so the bridge rejects
    grayscale (2D) and non-3-channel frames before it ever creates a writer.
    """
    b = _bridge(enable_commands=False)
    topic = "rt/test_arm/wrist/image_raw"

    # 2D grayscale, 4-channel RGBA, and a singleton channel are all malformed
    # for an rgb8 Image -> each must be dropped without publishing.
    b.publish_image("test_arm", "wrist", np.zeros((4, 6), dtype=np.uint8))
    b.publish_image("test_arm", "wrist", np.zeros((4, 6, 4), dtype=np.uint8))
    b.publish_image("test_arm", "wrist", np.zeros((4, 6, 1), dtype=np.uint8))
    assert not any(w.topic == topic for w in fake_cyclonedds["writers"])

    # A well-formed RGB frame afterwards still publishes -> the guard is
    # specific to malformed frames, not a blanket mute of the camera topic.
    b.publish_image("test_arm", "wrist", np.zeros((4, 6, 3), dtype=np.uint8))
    writer = next(w for w in fake_cyclonedds["writers"] if w.topic == topic)
    (msg,) = writer.samples
    assert (msg.height, msg.width, msg.encoding) == (4, 6, "rgb8")


def test_command_subscription_drives_send_action(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot)
    reader = next(r for r in fake_cyclonedds["readers"] if r.topic == "rt/test_arm/joint_command")
    reader.feed(_JointState(name=["a", "b"], position=[0.5, -0.5]))
    b._on_command(reader.take()[0])
    assert robot.sent_actions == [{"a": 0.5, "b": -0.5}]
    b.shutdown()


def test_empty_sample_is_skipped_silently(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot)
    b._on_command(_JointState(name=[], position=[]))
    assert robot.sent_actions == []
    b.shutdown()


def test_length_mismatch_is_ignored(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot)
    b._on_command(_JointState(name=["a", "b"], position=[0.5]))
    assert robot.sent_actions == []
    b.shutdown()


def test_read_only_when_disabled(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot, enable_commands=False)
    assert not [r for r in fake_cyclonedds["readers"] if r.topic.endswith("joint_command")]
    b.shutdown()


def test_pure_publisher_has_no_command_surface(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge()  # robot=None
    assert fake_cyclonedds["readers"] == []
    b.shutdown()


def test_shutdown_is_idempotent(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge(_FakeRobot())
    b.shutdown()
    assert b._poll_thread is None
    b.shutdown()


def test_missing_cyclonedds_raises_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    monkeypatch.setitem(sys.modules, "cyclonedds", None)
    from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

    with pytest.raises(ImportError):
        HardwareRtpsBridge(_FakeRobot())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DDS Security gate on the inbound command surface
# ---------------------------------------------------------------------------

_VALID_SECURITY = {
    "identity_ca": "file:/etc/dds/identity_ca.pem",
    "certificate": "file:/etc/dds/participant_cert.pem",
    "private_key": "file:/etc/dds/participant_key.pem",
    "governance": "file:/etc/dds/governance.p7s",
    "permissions": "file:/etc/dds/permissions.p7s",
}


def test_command_surface_refuses_without_security_or_optout(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # An enabled inbound command surface drives the physical arm, so without a
    # dds_security_config and without the explicit opt-out the bridge refuses.
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    with pytest.raises(ValueError, match="unsecured DDS graph"):
        _bridge(_FakeRobot())


def test_command_surface_allowed_with_insecure_optout(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", "yes")
    b = _bridge(_FakeRobot())
    assert b._enable_commands is True
    b.shutdown()


def test_command_surface_allowed_with_security_config_when_env_unset(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    b = _bridge(_FakeRobot(), dds_security_config=dict(_VALID_SECURITY))
    assert b._enable_commands is True
    b.shutdown()


def test_telemetry_only_bridge_is_not_gated(fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    # enable_commands=False is publish-only: no inbound surface, so no gate.
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    b = _bridge(_FakeRobot(), enable_commands=False)
    assert b._enable_commands is False
    b.shutdown()


def test_security_config_missing_required_key_raises(fake_cyclonedds: dict[str, Any]) -> None:
    incomplete = dict(_VALID_SECURITY)
    del incomplete["private_key"]
    with pytest.raises(ValueError, match="missing required keys"):
        _bridge(_FakeRobot(), dds_security_config=incomplete)


@pytest.mark.parametrize(
    "bad",
    [None, 0, False, b"file:/etc/dds/participant_key.pem", 1.5],
    ids=["none", "zero", "false", "bytes", "float"],
)
def test_a_non_string_credential_builds_no_participant_at_all(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch, bad: object
) -> None:
    # A credential the participant QoS cannot carry verbatim - it sets a property
    # per truthy credential, and stringifies what it does set - is refused before
    # any DDS state exists, rather than reaching DomainParticipant with the auth
    # plugin wired and no private key (falsy: dropped) or a private key spelled
    # "b'file:/etc/dds/participant_key.pem'" (truthy non-string: its repr).
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    cfg = dict(_VALID_SECURITY)
    cfg["private_key"] = bad  # type: ignore[assignment]
    before = len(fake_cyclonedds["participants"])
    with pytest.raises(ValueError, match="private_key"):
        _bridge(_FakeRobot(), dds_security_config=cfg)
    assert len(fake_cyclonedds["participants"]) == before


def test_a_supplied_permissions_ca_reaches_the_participant_qos(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The optional key's other half: absent it is not set (asserted below), and
    # supplied it is graded like a required one, so an accepted value lands.
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    cfg = dict(_VALID_SECURITY, permissions_ca="file:/etc/dds/permissions_ca.pem")
    b = _bridge(_FakeRobot(), dds_security_config=cfg)
    props = {p.name: p.value for p in fake_cyclonedds["participants"][-1].qos.policies}
    assert props["dds.sec.access.permissions_ca"] == "file:/etc/dds/permissions_ca.pem"
    b.shutdown()


def test_security_config_wires_plugins_and_credentials_into_participant_qos(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    b = _bridge(_FakeRobot(), dds_security_config=dict(_VALID_SECURITY))
    participant = fake_cyclonedds["participants"][-1]
    assert participant.qos is not None
    props = {p.name: p.value for p in participant.qos.policies}
    # Builtin DDS-Security plugins are wired automatically.
    assert props["dds.sec.auth.library.path"] == "dds_security_auth"
    assert props["dds.sec.crypto.library.path"] == "dds_security_crypto"
    assert props["dds.sec.access.library.path"] == "dds_security_ac"
    # Operator credentials are mapped to their dds.sec.* property names.
    assert props["dds.sec.auth.identity_ca"] == _VALID_SECURITY["identity_ca"]
    assert props["dds.sec.auth.identity_certificate"] == _VALID_SECURITY["certificate"]
    assert props["dds.sec.auth.private_key"] == _VALID_SECURITY["private_key"]
    assert props["dds.sec.access.governance"] == _VALID_SECURITY["governance"]
    assert props["dds.sec.access.permissions"] == _VALID_SECURITY["permissions"]
    # Optional permissions_ca absent -> property not set.
    assert "dds.sec.access.permissions_ca" not in props
    b.shutdown()


def test_telemetry_only_participant_has_no_security_qos(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge(enable_commands=False)  # no robot, no config
    assert fake_cyclonedds["participants"][-1].qos is None
    b.shutdown()


# ---------------------------------------------------------------------------
# Joint position bounds enforcement on the inbound command surface
# ---------------------------------------------------------------------------


def test_joint_limits_apply_in_range_command(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot, joint_limits={"a": (-1.0, 1.0), "b": (-1.0, 1.0)})
    b._on_command(_JointState(name=["a", "b"], position=[0.5, -0.5]))
    assert robot.sent_actions == [{"a": 0.5, "b": -0.5}]
    b.shutdown()


def test_joint_limits_reject_whole_command_when_one_joint_out_of_range(
    fake_cyclonedds: dict[str, Any],
) -> None:
    # One joint out of range -> the ENTIRE command is dropped (no partial apply).
    robot = _FakeRobot()
    b = _bridge(robot, joint_limits={"a": (-1.0, 1.0), "b": (-1.0, 1.0)})
    b._on_command(_JointState(name=["a", "b"], position=[0.5, 9.0]))
    assert robot.sent_actions == []
    b.shutdown()


def test_joint_limits_do_not_constrain_undeclared_joints(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot, joint_limits={"a": (-1.0, 1.0)})
    # "b" has no declared bound -> not constrained.
    b._on_command(_JointState(name=["a", "b"], position=[0.5, 100.0]))
    assert robot.sent_actions == [{"a": 0.5, "b": 100.0}]
    b.shutdown()


def test_invalid_joint_limits_raise_at_construction(fake_cyclonedds: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="min .* > max|must be a"):
        _bridge(_FakeRobot(), joint_limits={"a": (1.0, -1.0)})
    with pytest.raises(ValueError, match="must be a"):
        _bridge(_FakeRobot(), joint_limits={"a": 5.0})  # type: ignore[dict-item]


def test_start_poll_is_idempotent(fake_cyclonedds: dict[str, Any]) -> None:
    """A second start while the poll thread is alive spawns no second thread.

    The cyclonedds mirror of ``test_start_spin_is_idempotent``: two threads on
    one reader would each ``take()`` and split the command stream between them.
    """
    b = _bridge(_FakeRobot())  # enable_commands default -> poll thread started
    first = b._poll_thread
    assert first is not None and first.is_alive()

    b._start_poll()
    assert b._poll_thread is first

    b.shutdown()
    assert b._poll_thread is None


# --- per-robot topics (both transports) --------------------------------------


def test_each_robot_published_through_one_bridge_reaches_its_own_joint_states_topic(
    fake_cyclonedds: dict[str, Any],
) -> None:
    """``robot`` is a per-call argument, so it has to select the writer too.

    A bridge in telemetry-only mode is a pure publisher, and ``robot`` is the
    topic namespace the caller supplies per call - the shape
    ``SimEngine._publish_ros_telemetry`` drives when it loops over
    ``list_robots()``. Caching one writer for the whole bridge publishes every
    later robot's state on whichever robot happened to publish first, and the
    sample is not even wrong about itself: its ``frame_id`` names the robot it
    came from while the topic names another, so a subscriber to the second
    robot's ``joint_states`` sees no writer at all and the first robot's topic
    carries two arms interleaved. Nothing reports it - a write to a matched
    writer succeeds either way.
    """
    b = _bridge(enable_commands=False)
    for robot, positions in (("arm_a", [0.1]), ("arm_b", [7.7]), ("arm_a", [0.2])):
        b.publish_joint_states(robot, ["j0"], positions)

    by_topic: dict[str, list[Any]] = {}
    for writer in fake_cyclonedds["writers"]:
        by_topic.setdefault(writer.topic, []).extend(writer.samples)

    assert sorted(by_topic) == ["rt/arm_a/joint_states", "rt/arm_b/joint_states"]
    assert [s.position for s in by_topic["rt/arm_a/joint_states"]] == [[0.1], [0.2]]
    assert [s.position for s in by_topic["rt/arm_b/joint_states"]] == [[7.7]]
    # Two robots, three publishes: the writers are cached per robot, not per call.
    assert len(fake_cyclonedds["writers"]) == 2
    # Every sample's own frame_id agrees with the topic it was written to.
    for topic, samples in by_topic.items():
        for sample in samples:
            assert f"rt/{sample.header.frame_id}/joint_states" == topic


def test_the_rtps_and_rclpy_transports_advertise_the_same_per_robot_topics(
    fake_cyclonedds: dict[str, Any],
) -> None:
    """The two transports are byte-identical on the graph only if both key on ``robot``.

    ``RosTelemetryBase`` exists so the topic names live in one place, but each
    subclass owns its own publisher cache - so the cache is where the two can
    still disagree about how many topics one call sequence advertises. This
    drives the same sequence through both and compares what each would put on
    the graph. The rclpy side is read from its real ``_joint_publisher`` /
    ``_image_publisher``, bound to a stub, so no sourced ROS 2 distro is needed.
    """
    from types import SimpleNamespace

    from strands_robots.ros_telemetry import RosTelemetryBridge
    from strands_robots.rtps.mangling import dds_topic_name

    calls = (("arm_a", "wrist"), ("arm_b", "wrist"))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    bridge = _bridge(enable_commands=False)
    for robot, camera in calls:
        bridge.publish_joint_states(robot, ["j0"], [0.0])
        bridge.publish_image(robot, camera, frame)
    rtps_topics = sorted(w.topic for w in fake_cyclonedds["writers"])

    created: list[str] = []

    class _Node:
        def create_publisher(self, _msg_type: Any, topic: str, _depth: int) -> object:
            created.append(topic)
            return object()

    stub = SimpleNamespace(
        _joint_pubs={},
        _image_pubs={},
        _node=_Node(),
        _JointState=object(),
        _Image=object(),
        _qos_depth=10,
        joint_states_topic=RosTelemetryBridge.joint_states_topic,
        image_topic=RosTelemetryBridge.image_topic,
    )
    # The two cache methods read only the attributes above, so binding them to
    # the stub grades the real rclpy codepath without a rclpy install.
    as_bridge = cast(RosTelemetryBridge, stub)
    for robot, camera in calls:
        RosTelemetryBridge._joint_publisher(as_bridge, robot)
        RosTelemetryBridge._image_publisher(as_bridge, robot, camera)

    assert created, "premise: the rclpy transport advertised nothing"
    assert rtps_topics == sorted(dds_topic_name(topic) for topic in created)


#: Call sequences in which the caller-supplied names and the topics they select
#: do not correspond one-to-one, which is the only place a cache keyed on the
#: names can disagree with one keyed on the topic. Both are reachable: a scene
#: may hold cameras named ``arm0/wrist`` and ``arm0__wrist`` (both forms are
#: documented as legal, and ``camera_schema_key`` names that very pair as the
#: collision its dataset-side guard exists for), and ``robot``/``camera`` may
#: each contain the ``/`` the former cache key joined them with.
_AMBIGUOUS_IMAGE_CALLS = (
    pytest.param(
        [("so101", "arm0/wrist"), ("so101", "arm0__wrist"), ("so101", "default")],
        id="two_camera_spellings_naming_one_topic",
    ),
    pytest.param(
        [("arm", "wrist/rgb"), ("arm/wrist", "rgb")],
        id="one_slash_joined_key_naming_two_topics",
    ),
)


def _rclpy_cache_stub() -> Any:
    """A ``RosTelemetryBridge`` whose publishers record the topic they were made for.

    Subclassed with the rclpy-dependent constructor replaced, so the real
    ``publish_joint_states`` / ``publish_image`` and their real publisher caches
    run with no ROS 2 distro installed.
    """
    from types import SimpleNamespace

    from strands_robots.ros_telemetry import RosTelemetryBridge

    class _Pub:
        def __init__(self, topic: str) -> None:
            self.topic = topic
            self.samples: list[Any] = []

        def publish(self, msg: Any) -> None:
            self.samples.append(msg)

    class _Node:
        def __init__(self) -> None:
            self.pubs: list[_Pub] = []

        def create_publisher(self, _msg_type: Any, topic: str, _depth: int) -> _Pub:
            pub = _Pub(topic)
            self.pubs.append(pub)
            return pub

    class _StubBridge(RosTelemetryBridge):
        def __init__(self) -> None:
            self._node = _Node()
            self._joint_pubs = {}
            self._image_pubs = {}
            self._qos_depth = 10
            self._JointState = lambda: SimpleNamespace(header=_Header(), name=[], position=[])
            self._Image = lambda: SimpleNamespace(header=_Header())

        def _now(self) -> Any:
            return None

    return _StubBridge()


def _samples_per_topic(entities: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entity in entities:
        counts[entity.topic] = counts.get(entity.topic, 0) + len(entity.samples)
    return counts


@pytest.mark.parametrize("calls", _AMBIGUOUS_IMAGE_CALLS)
def test_both_transports_hold_one_image_publisher_per_topic(
    fake_cyclonedds: dict[str, Any], calls: list[tuple[str, str]]
) -> None:
    """A publisher is identified by its topic, so the cache must be keyed on it.

    Keyed on the caller's spelling instead, the name -> topic map's two failures
    each produced a wrong graph. It is not injective, so two camera spellings of
    one topic advertised two publishers on it - one bridge appearing twice in
    ``ros2 topic info`` for one camera. And the key joined ``robot`` and
    ``camera`` with ``/``, a character both may contain, so two pairs naming two
    different topics shared one key: the second caller was handed the first's
    publisher and its frames went out on a topic it never named, silently,
    because DDS matching is by topic name and the reader it expected never
    appeared.

    Both claims reduce to one measurement, made on both transports so they
    cannot come to disagree: every frame lands on the topic its own call names,
    and there is exactly one publisher per distinct topic named.
    """
    from strands_robots.ros_telemetry import RosTelemetryBridge
    from strands_robots.rtps.mangling import dds_topic_name

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    expected = {RosTelemetryBridge.image_topic(robot, camera): 0 for robot, camera in calls}
    for robot, camera in calls:
        expected[RosTelemetryBridge.image_topic(robot, camera)] += 1
    assert len(expected) < len(calls) or len({f"{r}/{c}" for r, c in calls}) < len(calls), (
        "premise: this sequence must exercise a name/topic mismatch"
    )

    bridge = _bridge(enable_commands=False)
    for robot, camera in calls:
        bridge.publish_image(robot, camera, frame)
    rtps = _samples_per_topic(fake_cyclonedds["writers"])
    assert rtps == {dds_topic_name(topic): n for topic, n in expected.items()}
    assert len(fake_cyclonedds["writers"]) == len(expected)

    stub = _rclpy_cache_stub()
    for robot, camera in calls:
        stub.publish_image(robot, camera, frame)
    assert _samples_per_topic(stub._node.pubs) == expected
    assert len(stub._node.pubs) == len(expected)


def test_both_transports_hold_one_joint_publisher_per_topic(fake_cyclonedds: dict[str, Any]) -> None:
    """Two robot names selecting one ``joint_states`` topic share its publisher.

    The joint cache has only the non-injectivity half of the problem above - it
    keys on one name, so there is no join to be ambiguous - but the consequence
    is the same duplicated advertisement, and the fix is the same key.
    """
    from strands_robots.ros_telemetry import RosTelemetryBridge
    from strands_robots.rtps.mangling import dds_topic_name

    calls = ["front cam", "front-cam"]
    topic = RosTelemetryBridge.joint_states_topic(calls[0])
    assert {RosTelemetryBridge.joint_states_topic(r) for r in calls} == {topic}, (
        "premise: both names must select one topic"
    )

    bridge = _bridge(enable_commands=False)
    for robot in calls:
        bridge.publish_joint_states(robot, ["j0"], [0.0])
    assert _samples_per_topic(fake_cyclonedds["writers"]) == {dds_topic_name(topic): len(calls)}
    assert len(fake_cyclonedds["writers"]) == 1

    stub = _rclpy_cache_stub()
    for robot in calls:
        stub.publish_joint_states(robot, ["j0"], [0.0])
    assert _samples_per_topic(stub._node.pubs) == {topic: len(calls)}
    assert len(stub._node.pubs) == 1


def test_shutdown_drops_every_robots_joint_writer(fake_cyclonedds: dict[str, Any]) -> None:
    """Shutdown releases the DDS entities for all robots, not just one.

    The python binding has no explicit ``close()``, so dropping the references
    is what lets cyclonedds reclaim the writers; a writer left behind keeps this
    participant advertising a robot the bridge no longer publishes.
    """
    b = _bridge(enable_commands=False)
    b.publish_joint_states("arm_a", ["j0"], [0.1])
    b.publish_joint_states("arm_b", ["j0"], [0.2])
    assert len(fake_cyclonedds["writers"]) == 2

    b.shutdown()
    assert b._joint_writers == {}
    assert b._image_writers == {}


# --- the command namespace is the one caller-supplied name in a topic --------

#: ``command_robot_name`` values that cannot name a topic segment. A truthy one
#: reached the sanitiser's ``re.sub`` and raised ``TypeError`` naming no
#: parameter; a falsy one was filtered by the ``command_robot_name or <derived>``
#: default and never raised, so the bridge read commands under a namespace the
#: caller had not asked for. Both happened past the ``DomainParticipant``.
_UNNAMEABLE_COMMAND_NAMESPACES: list[Any] = [7, ["left_arm"], 0, []]


@pytest.mark.parametrize("name", _UNNAMEABLE_COMMAND_NAMESPACES, ids=repr)
def test_a_command_namespace_that_cannot_name_a_topic_builds_no_participant(
    fake_cyclonedds: dict[str, Any], name: Any
) -> None:
    """The refusal names the parameter and leaves no DDS state behind.

    ``__init__`` raising returns no bridge, so a participant built before the
    refusal has no ``shutdown`` that can reach it. The namespace is therefore
    graded where ``domain_id``, ``poll_period`` and ``enable_commands`` are.
    """
    with pytest.raises(ValueError) as excinfo:
        _bridge(_FakeRobot(), command_robot_name=name)

    assert "'command_robot_name'" in str(excinfo.value)
    assert fake_cyclonedds["participants"] == []
    assert fake_cyclonedds["readers"] == []


def test_a_command_namespace_is_graded_before_the_cyclonedds_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same mistake reports identically on an install without the extra.

    With no ``cyclonedds`` importable, a value that cannot name a topic must
    still be refused for the value - not reported as a missing dependency - which
    is what the other three constructor guards already promise.
    """
    monkeypatch.setitem(sys.modules, "cyclonedds", None)
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)

    with pytest.raises(ValueError, match="'command_robot_name'"):
        _bridge(_FakeRobot(), command_robot_name=7)


@pytest.mark.parametrize(
    ("name", "expected_topic"),
    [
        (None, "rt/test_arm/joint_command"),
        ("", "rt/test_arm/joint_command"),
        ("left_arm", "rt/left_arm/joint_command"),
    ],
    ids=["derive/None", "derive/empty", "override"],
)
def test_an_accepted_command_namespace_selects_the_topic_it_names(
    fake_cyclonedds: dict[str, Any], name: Any, expected_topic: str
) -> None:
    """``None`` and ``""`` derive the bound robot's name; a string overrides it."""
    b = _bridge(_FakeRobot(), command_robot_name=name)
    assert [r.topic for r in fake_cyclonedds["readers"]] == [expected_topic]
    b.shutdown()
