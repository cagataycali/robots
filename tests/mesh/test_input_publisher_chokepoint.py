"""InputPublisher must publish teleop frames through Mesh.publish().

Mesh.publish() is documented as the single publish chokepoint so a future
audit / telemetry / compression hook lands in one place. Every publisher
(sensors, state, presence, cameras, commands) routes through it. The teleop
input stream -- remote joint actuation, the most safety-critical publish path
-- must not be the lone exception that bypasses the chokepoint by calling the
module-level session.put() directly. These tests pin that contract: a hook on
Mesh.publish() observes outbound teleop frames.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from strands_robots.mesh.input import InputPublisher


class _FakeMesh:
    """Minimal mesh exposing the publish chokepoint and a peer_id.

    Records every (key, payload) handed to publish() so the test can assert
    teleop frames went through the chokepoint rather than around it.
    """

    def __init__(self, peer_id: str = "pub-chokepoint") -> None:
        self.peer_id = peer_id
        self.alive = True
        self.published: list[tuple[str, dict[str, Any]]] = []

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        self.published.append((key, payload))


def _run_publisher(mesh: _FakeMesh, teleop: Any, hz: float = 200.0) -> dict[str, Any]:
    pub = InputPublisher(mesh, teleop, device_name="leader", method="arm", hz=hz)  # type: ignore[arg-type]
    pub.start()
    # Spin until at least one frame is published or a short deadline elapses.
    deadline = time.time() + 1.0
    while not mesh.published and time.time() < deadline:
        time.sleep(0.005)
    return pub.stop()


def test_publisher_routes_frames_through_mesh_publish() -> None:
    """Frames reach Mesh.publish() on the canonical input topic."""
    teleop = MagicMock()
    teleop.get_action.return_value = {"j0": 0.1, "j1": -0.2}

    mesh = _FakeMesh(peer_id="pub-1")
    stats = _run_publisher(mesh, teleop)

    assert stats["frames"] > 0
    # The chokepoint -- not the raw module put() -- carried the frames.
    assert mesh.published, "teleop frames bypassed Mesh.publish() chokepoint"
    topic, payload = mesh.published[0]
    assert topic == "strands/pub-1/input/leader"
    assert payload["peer_id"] == "pub-1"
    assert payload["device"] == "leader"
    assert payload["action"] == {"j0": 0.1, "j1": -0.2}


def test_publisher_frame_count_matches_chokepoint_calls() -> None:
    """Every counted frame corresponds to a chokepoint publish (no leakage)."""
    teleop = MagicMock()
    teleop.get_action.return_value = {"shoulder_pan": 0.0}

    mesh = _FakeMesh(peer_id="pub-2")
    pub = InputPublisher(mesh, teleop, device_name="gamepad", method="gamepad", hz=200.0)  # type: ignore[arg-type]
    pub.start()
    deadline = time.time() + 1.0
    while len(mesh.published) < 3 and time.time() < deadline:
        time.sleep(0.005)
    stats = pub.stop()

    # Frames the publisher claims it sent all went through the chokepoint.
    assert stats["frames"] == len(mesh.published)
    assert all(k == "strands/pub-2/input/gamepad" for k, _ in mesh.published)


def test_normalize_action_dict_floats_each_value() -> None:
    """A dict action keeps its keys, coercing every value to float."""
    import numpy as np

    out = InputPublisher._normalize_action({"shoulder.pos": np.float32(1.5), "j0": 2})
    assert out == {"shoulder.pos": 1.5, "j0": 2.0}


def test_normalize_action_array_becomes_positional_keys() -> None:
    """A 1-D array action becomes positional ``jN`` keys."""
    import numpy as np

    assert InputPublisher._normalize_action(np.array([1.0, 2.0, 3.0])) == {
        "j0": 1.0,
        "j1": 2.0,
        "j2": 3.0,
    }


def test_normalize_action_scalar_does_not_crash() -> None:
    """A numpy/torch scalar or 0-d array exposes ``tolist()`` that returns a
    bare Python number, not a list. Enumerating it raises
    ``'float' object is not iterable``; a 1-DOF leader must not crash the
    input stream. Such a value normalizes to the single-DOF ``{"raw": ...}``
    shape, matching the plain-Python-scalar fallback.
    """
    import numpy as np

    assert InputPublisher._normalize_action(2.0) == {"raw": 2.0}
    assert InputPublisher._normalize_action(np.float32(1.5)) == {"raw": 1.5}
    assert InputPublisher._normalize_action(np.array(1.5)) == {"raw": 1.5}
