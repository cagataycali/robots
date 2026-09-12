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
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from strands_robots.hardware_robot import _build_camera_config
from strands_robots.utils import camera_token_error

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
