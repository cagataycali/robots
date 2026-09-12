"""A camera's name is held to one alphabet at every door that accepts one.

A camera's name in a ``cameras`` mapping is the identity every downstream
consumer keys that camera's frames by, and each of them reserves punctuation of
its own: the mesh publishes the frames on ``strands/<peer_id>/camera/<name>``,
the IoT offload joins the name into the S3 object key, and a recording writes it
as the ``observation.images.<name>`` dataset feature key. ``lerobot_teleoperate``
already refused a name that is not a bare token, because it renders one into the
nested ``--robot.cameras`` argv; the ``Robot`` factory accepted any name at all
and the two doors disagreed. These tests pin that they now share one rule
(:func:`~strands_robots.utils.camera_token_error`), and the wire outcome that
makes the rule worth having.

The third door is ``add_camera``, and it reaches the same consumers: a sim robot
on the mesh publishes its camera frames through the same
``strands/<peer_id>/camera/<name>`` topic and the same S3 offload. It was held to
:func:`~strands_robots.utils.entity_name_error` only, so a name none of those
consumers can carry registered under ``status="success"`` - measured on one
``create_world``, ``add_camera`` accepted ``'..'``, ``'sub/../etc'``, ``'a b'``,
``'cam#1'``, ``'*'``, ``'**'``, ``'a//b'``, ``'/lead'`` and ``'trail/'``.

That door takes one more thing than the hardware doors do, and exactly one: a
robot scope. ``add_robot`` namespaces everything it spawns under ``<robot>/``
(``SimRobot.namespace``), so ``add_camera("alice/wrist_cam", ...)`` is how a
wrist camera is scoped to the robot ``alice``, and
:meth:`~strands_robots.mesh.core.Mesh._publish_sim_cameras` strips exactly that
one prefix before publishing - so a one-level name reaches the topic as the bare
token the hardware doors require, which
``test_the_namespace_strip_leaves_a_scoped_sim_camera_at_one_topic_level``
measures. A second level survives the strip and is structure again, so
:func:`~strands_robots.utils.scoped_camera_name_error` allows one and refuses
two. Holding the sim door to the hardware doors' single-token rule instead would
refuse ``alice/wrist_cam``, which is a documented and tested way to name a
multi-robot scene's cameras.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from strands_robots.hardware_robot import _build_camera_config
from strands_robots.utils import camera_name_error, camera_token_error

_teleop = importlib.import_module("strands_robots.tools.lerobot_teleoperate")
_iot = importlib.import_module("strands_robots.mesh.transport.iot_transport")
_offload = importlib.import_module("strands_robots.mesh.iot.camera_offload")

# Each name, and the phrase both doors must answer it with. A name is refused
# for what it would do downstream, so the table is one row per reserved
# character rather than one file per spelling.
UNUSABLE = [
    pytest.param("wrist/ref", "bare token", id="topic-level-that-spells-the-pointer-tail"),
    pytest.param("front/left", "bare token", id="topic-level"),
    pytest.param("*", "bare token", id="zenoh-wildcard"),
    pytest.param("**", "bare token", id="zenoh-multi-wildcard"),
    pytest.param("..", "bare token", id="parent-of-the-s3-prefix"),
    pytest.param("wrist.rgb", "bare token", id="dataset-feature-key-separator"),
    pytest.param("front,wrist", "bare token", id="argv-dict-separator"),
    pytest.param("front wrist", "bare token", id="whitespace"),
    pytest.param("front\n--robot.port=/dev/x", "bare token", id="newline"),
    pytest.param("", "non-empty string", id="empty"),
    pytest.param(7, "non-empty string", id="not-a-str"),
]

USABLE = ["wrist", "front_cam", "top", "cam-2", "0"]


@pytest.mark.parametrize(("name", "phrase"), UNUSABLE)
def test_both_doors_refuse_a_name_no_consumer_could_carry(name, phrase):
    """The factory and the teleop tool refuse the same name for the same reason."""
    with pytest.raises(ValueError, match=phrase):
        _build_camera_config(name, {"index_or_path": 0})

    tool_error = _teleop._camera_map_error({name: {"index_or_path": 0}})
    assert tool_error is not None and phrase in tool_error


@pytest.mark.parametrize("name", USABLE)
def test_both_doors_accept_a_bare_token(name):
    """A name every consumer can carry still reaches a built camera config."""
    assert camera_token_error("Robot(cameras=...)", "camera name", name) is None
    built = _build_camera_config(name, {"index_or_path": 0})
    assert (built.width, built.height) == (640, 480)
    assert _teleop._camera_map_error({name: {"index_or_path": 0}}) is None


def test_the_frame_topic_a_refused_name_would_publish_on_reads_as_a_pointer():
    """Why the rule: a name spelling an extra level turns a frame into a 'pointer'.

    The IoT transport drops camera frames rather than paying WAN for a base64
    JPEG, and exempts the small S3 pointer published on
    ``strands/<peer>/camera/<name>/ref``. It grants that exemption on the
    topic's shape, so a camera named ``wrist/ref`` publishes its *inline* frame
    on exactly the pointer's shape and the drop lets the whole frame through.
    """
    from strands_robots.mesh import Mesh

    name = "wrist/ref"
    inner = SimpleNamespace(is_connected=True, name="so101", config=SimpleNamespace(cameras={name: {}}))
    mesh = Mesh(SimpleNamespace(tool_name_str="so101", robot=inner), peer_id="rover-01")
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._encode_and_publish_frames({name: frame}, [name])

    topic, payload = mock_put.call_args[0]
    assert topic == "strands/rover-01/camera/wrist/ref"
    assert len(payload["data"]) > 1000  # an inline frame, not a few-hundred-byte pointer
    assert _iot._is_camera_ref(topic) is True
    assert _iot._should_drop(topic) is False
    # The door is what keeps that name off the wire in the first place.
    assert camera_token_error("Robot(cameras=...)", "camera name", name) is not None


def test_the_s3_key_a_refused_name_would_write_leaves_the_peer_prefix():
    """Why the rule: the offload joins the name into the object key unchanged."""
    offloader = _offload.CameraOffloader(bucket="fleet-frames", prefix="frames")
    assert offloader.s3_key_for("rover-01", "wrist", 123) == "frames/rover-01/wrist/123.jpg"
    assert offloader.s3_key_for("rover-01", "../..", 123) == "frames/rover-01/../../123.jpg"
    assert camera_token_error("Robot(cameras=...)", "camera name", "../..") is not None


# --------------------------------------------------------------------------- #
# The sim door: the same alphabet, plus one robot scope                       #
# --------------------------------------------------------------------------- #

#: Names ``add_camera`` registered under ``status="success"`` before this rule.
#: One row per reason a consumer cannot carry the name, not one per spelling.
SIM_UNUSABLE = [
    pytest.param("..", id="parent-of-the-s3-prefix"),
    pytest.param("sub/../etc", id="traversal-inside-a-scope"),
    pytest.param("*", id="zenoh-wildcard"),
    pytest.param("**", id="zenoh-multi-wildcard"),
    pytest.param("a b", id="whitespace"),
    pytest.param("cam#1", id="punctuation"),
    pytest.param("cam.1", id="dataset-feature-key-separator"),
    pytest.param("a//b", id="empty-scope"),
    pytest.param("/lead", id="leading-separator"),
    pytest.param("trail/", id="trailing-separator"),
    pytest.param("alice/wrist/left", id="two-levels-outlive-the-namespace-strip"),
]

#: Names that must keep registering. ``alice/wrist_cam`` and ``arm0/wrist`` are
#: the scoped form ``tests/simulation/mujoco/test_recording_camera_scoping.py``,
#: ``tests/simulation/mujoco/test_render_all_multicamera.py`` and
#: ``tests_integ/simulation/test_multi_robot_tasks.py`` already name cameras with.
SIM_USABLE = ["wrist", "front_cam", "cam-2", "0cam", "alice/wrist_cam", "arm0/wrist"]


@pytest.fixture
def sim():
    """An empty MuJoCo world. ``add_camera`` compiles the spec but renders nothing."""
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    s = Simulation(tool_name="test_camera_name_alphabet_sim", mesh=False)
    assert s.create_world()["status"] == "success"
    yield s
    s.cleanup()


@pytest.mark.parametrize("name", SIM_UNUSABLE)
def test_the_sim_door_refuses_a_name_no_consumer_could_carry(sim, name):
    """Refused, and nothing half-registered under the name."""
    result = sim.add_camera(name=name, position=[0.6, 0.6, 0.5], target=[0.0, 0.0, 0.1])
    assert result["status"] == "error", (name, result)
    assert name not in sim._world.cameras


@pytest.mark.parametrize("name", SIM_UNUSABLE)
def test_every_backend_refuses_it_for_the_same_reason(name):
    """The rule is the shared one, so the two routing postures agree on the name.

    ``routes_free_camera_tokens`` selects whether a backend also refuses the
    free-camera routing tokens; it must not select which alphabet a name is held
    to, or the Isaac backend would keep the older, wider one.
    """
    routing = camera_name_error("add_camera", "name", name, routes_free_camera_tokens=True)
    lookup = camera_name_error("add_camera", "name", name, routes_free_camera_tokens=False)
    assert routing is not None and lookup == routing
    assert "cannot key a camera's frames" in routing


@pytest.mark.parametrize("name", SIM_USABLE)
def test_the_sim_door_accepts_a_token_optionally_scoped_to_one_robot(sim, name):
    """A camera a multi-robot scene names this way still registers and is listed."""
    assert camera_name_error("add_camera", "name", name, routes_free_camera_tokens=True) is None
    result = sim.add_camera(name=name, position=[0.6, 0.6, 0.5], target=[0.0, 0.0, 0.1])
    assert result["status"] == "success", (name, result)
    assert name in sim._world.cameras


def test_a_reserved_routing_token_is_still_answered_as_reserved(sim):
    """The added clause is last, so it does not restate an earlier refusal.

    ``'default'`` is a perfectly good token; what is wrong with it is that this
    backend's ``render`` resolves it to the free camera, and that is the message
    a caller has to read to act.
    """
    message = camera_name_error("add_camera", "name", "default", routes_free_camera_tokens=True)
    assert message is not None and "reserved" in message
    assert camera_name_error("add_camera", "name", "default", routes_free_camera_tokens=False) is None


def test_the_namespace_strip_leaves_a_scoped_sim_camera_at_one_topic_level(sim, monkeypatch):
    """Why one level and not a path: the mesh removes exactly one prefix.

    ``_publish_sim_cameras`` strips ``SimRobot.namespace`` (``"<robot>/"``) from
    each compiled camera name before handing the short name to
    ``_encode_and_publish_frames``, so a one-level scoped name is published on a
    single topic level - the shape ``strands/*/camera/*`` matches. A second level
    would outlive the strip, which is why the door refuses it.
    """
    import mujoco

    from strands_robots.mesh import Mesh

    assert (
        sim.add_camera(name="alice/wrist_cam", position=[0.6, 0.6, 0.5], target=[0.0, 0.0, 0.1])["status"] == "success"
    )

    class _FakeRenderer:
        """Stands in for the GL renderer: this test grades the name, not the pixels."""

        def __init__(self, model, height=480, width=640):
            self._shape = (height, width, 3)

        def update_scene(self, data, camera=0):
            pass

        def render(self):
            return np.full(self._shape, 128, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", _FakeRenderer)
    robot = SimpleNamespace(tool_name_str="alice", name="alice", namespace="alice/", _world=sim._world)
    mesh = Mesh(robot, peer_id="rover-01")

    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._publish_sim_cameras()

    topics = [call.args[0] for call in mock_put.call_args_list]
    assert topics == ["strands/rover-01/camera/wrist_cam"]
    assert camera_token_error("Robot(cameras=...)", "camera name", "wrist_cam") is None
    # The level the strip cannot remove is the one the door refuses.
    assert camera_name_error("add_camera", "name", "alice/wrist/left", routes_free_camera_tokens=True) is not None
