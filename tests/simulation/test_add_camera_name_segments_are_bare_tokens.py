# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: ``add_camera`` holds each segment of a name to the bare-token rule.

A scene camera's name reaches the same consumers a hardware camera's does - the
mesh publishes its frames on ``strands/<peer_id>/camera/<name>``, the IoT offload
joins it into the S3 object key, a recording writes it as the
``observation.images.<name>`` dataset feature key - and
:func:`~strands_robots.utils.camera_token_error` hardened the hardware door
against a name those consumers read as structure. Every backend's ``add_camera``
stayed on the older rule, which refuses only a name that cannot be a registry key
at all. Measured on MuJoCo, one ``create_world`` then one ``add_camera`` per name:
``'a/b'``, ``'..'``, ``'sub/../etc'``, ``'a b'``, ``'cam#1'`` and ``'**'`` each
registered under ``status="success"``, and only ``''`` was refused.

The hardware rule cannot be applied verbatim, because the simulation itself writes
a ``/`` into a camera's name: a robot's cameras are registered under
``<namespace>/<cam>`` (``arm0/wrist``), ``render_all`` resolves a short name
against that form, and :func:`~strands_robots.utils.camera_schema_key` collapses
the separator for the dataset column. So the scene rule is
:func:`~strands_robots.utils.camera_segments_error`: every ``/``-separated segment
is a bare token, on the one alphabet :data:`~strands_robots.utils._CAMERA_TOKEN`
both doors read. Refusing ``/`` outright would refuse a shape three documented
surfaces produce, which is why these tests pin the namespaced form as accepted
beside the refusals.

Every backend is graded here, and each half runs where the optional solver is
absent: MuJoCo compiles the spec and renders nothing, and the Newton and Isaac
guards run before the method touches a solver or a stage, so an unbound call with
a small stand-in for ``self`` reaches them (the pattern
``tests/simulation/test_reserved_camera_name_at_creation.py`` uses).
"""

from __future__ import annotations

import threading
import types
from typing import Any, cast

import pytest

from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import camera_name_error, camera_segments_error

#: One row per reserved character, with the consumer that reads it as structure.
UNUSABLE = [
    pytest.param("..", id="parent-of-the-s3-prefix"),
    pytest.param("sub/../etc", id="traversal-inside-a-namespace"),
    pytest.param("**", id="zenoh-multi-wildcard"),
    pytest.param("*", id="zenoh-wildcard"),
    pytest.param("a b", id="whitespace"),
    pytest.param("cam#1", id="mjcf-and-usd-punctuation"),
    pytest.param("wrist.rgb", id="dataset-feature-key-separator"),
    pytest.param("/wrist", id="empty-leading-segment"),
    pytest.param("arm0/", id="empty-trailing-segment"),
    pytest.param("a//b", id="empty-middle-segment"),
    pytest.param("front\nwrist", id="newline"),
]

#: Names every consumer can carry, including the namespaced form the simulation
#: writes and the two shapes ``docs/simulation/world-building.md`` documents.
USABLE = ["wrist", "cam-1", "front_cam", "0", "arm0/wrist", "arm2/wrist", "so101/wrist_cam"]


class TestTheDomain:
    """The rule, read directly and through the composed camera name rule."""

    @pytest.mark.parametrize("name", UNUSABLE)
    def test_a_segment_that_is_not_a_bare_token_is_refused(self, name: str) -> None:
        err = camera_segments_error("add_camera", "name", name)
        assert err is not None, name
        assert err.startswith("add_camera: name=")
        assert "bare tokens joined by '/'" in err

    @pytest.mark.parametrize("name", USABLE)
    def test_bare_tokens_joined_by_a_slash_are_accepted(self, name: str) -> None:
        assert camera_segments_error("add_camera", "name", name) is None

    @pytest.mark.parametrize("routes", [True, False])
    @pytest.mark.parametrize("name", UNUSABLE)
    def test_the_composed_rule_refuses_it_on_every_backend_posture(self, name: str, routes: bool) -> None:
        """Both ``routes_free_camera_tokens`` postures reach this rule - it is not gated on one."""
        err = camera_name_error("add_camera", "name", name, routes_free_camera_tokens=routes)
        assert err is not None and "bare tokens" in err

    @pytest.mark.parametrize("routes", [True, False])
    @pytest.mark.parametrize("name", USABLE)
    def test_the_composed_rule_still_accepts_a_usable_name(self, name: str, routes: bool) -> None:
        assert camera_name_error("add_camera", "name", name, routes_free_camera_tokens=routes) is None

    def test_the_addressability_refusal_still_comes_first(self) -> None:
        """A value that is not a ``str`` is refused by ``entity_name_error``, so this rule never reads it."""
        err = camera_name_error("add_camera", "name", 7, routes_free_camera_tokens=True)
        assert err is not None and "bare tokens" not in err

    def test_the_reserved_refusal_is_still_reached_for_a_token(self) -> None:
        """A routing token is a bare token, so the reserved rule stays the one that refuses it."""
        err = camera_name_error("add_camera", "name", "default", routes_free_camera_tokens=True)
        assert err is not None and "reserved" in err

    def test_the_message_is_ascii(self) -> None:
        """Project rule: the rule's own text is plain ASCII; the ``repr`` of the refused value is the caller's."""
        for name in ("a b", "**", "..", "cam#1"):
            err = camera_segments_error("add_camera", "name", name)
            assert err is not None
            err.encode("ascii")


# --------------------------------------------------------------------------- #
# What a refused name would have done downstream                              #
# --------------------------------------------------------------------------- #


def test_the_topic_a_refused_name_would_publish_on_is_a_wildcard() -> None:
    """Why the rule: the publisher joins the raw name into the topic, so ``**`` is a Zenoh wildcard.

    A ``put`` on a wildcard key is routed by intersection, so the frame reaches
    every peer subscribed to any camera rather than the one asking for this one.
    The sim publisher strips a robot's namespace and hands the rest to the same
    encoder the hardware path uses, which is the seam graded here.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np

    from strands_robots.mesh import Mesh

    name = "**"
    inner = SimpleNamespace(is_connected=True, name="rover", config=SimpleNamespace(cameras={name: {}}))
    mesh = Mesh(SimpleNamespace(tool_name_str="rover", robot=inner), peer_id="rover01")
    frame = np.full((48, 64, 3), 128, dtype=np.uint8)

    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._encode_and_publish_frames({name: frame}, [name])

    topic, _payload = mock_put.call_args[0]
    assert topic == "strands/rover01/camera/**"
    # The door is what keeps that name out of the scene in the first place.
    assert camera_segments_error("add_camera", "name", name) is not None


def test_the_s3_key_a_refused_name_would_write_leaves_the_peer_prefix() -> None:
    """Why the rule: the offload joins the name into the object key unchanged."""
    from strands_robots.mesh.iot.camera_offload import CameraOffloader

    offloader = CameraOffloader(bucket="fleet-frames", prefix="frames")
    assert offloader.s3_key_for("rover01", "arm0/wrist", 123) == "frames/rover01/arm0/wrist/123.jpg"
    assert offloader.s3_key_for("rover01", "../../etc/passwd", 123) == "frames/rover01/../../etc/passwd/123.jpg"
    assert camera_segments_error("add_camera", "name", "../../etc/passwd") is not None


# --------------------------------------------------------------------------- #
# MuJoCo                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sim():
    pytest.importorskip("mujoco")
    from strands_robots.simulation import Simulation

    s = Simulation()
    s.create_world()
    yield s
    s.destroy()


class TestMujocoAddCamera:
    @pytest.mark.parametrize("name", UNUSABLE)
    def test_an_unusable_name_is_refused(self, sim, name: str) -> None:
        result = sim.add_camera(name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "error", (name, result)
        assert "bare tokens" in result["content"][0]["text"]

    @pytest.mark.parametrize("name", UNUSABLE)
    def test_the_refusal_registers_nothing(self, sim, name: str) -> None:
        before = dict(sim._world.cameras)
        sim.add_camera(name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert sim._world.cameras == before

    @pytest.mark.parametrize("name", USABLE)
    def test_a_usable_name_still_registers(self, sim, name: str) -> None:
        result = sim.add_camera(name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "success", (name, result)
        assert name in sim._world.cameras

    def test_a_namespaced_camera_compiles_under_the_name_it_was_given(self, sim) -> None:
        """The reason ``/`` stays admitted: the namespaced form is a first-class scene camera.

        ``tests/simulation/mujoco/test_render_all_multicamera.py`` pins that
        ``render_all`` resolves the short name onto it; this pins the half that
        belongs to the door - the name is accepted and the compiled model
        carries it, so the camera is reachable through the API that created it.
        """
        import mujoco

        assert (
            sim.add_camera(name="arm0/wrist", position=[0.4, 0.0, 0.5], target=[0.0, 0.0, 0.2])["status"] == "success"
        )
        assert mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_CAMERA, "arm0/wrist") >= 0


# --------------------------------------------------------------------------- #
# Newton                                                                      #
# --------------------------------------------------------------------------- #


def _newton_stub() -> types.SimpleNamespace:
    """A stand-in for ``self`` carrying only what ``add_camera`` reads before the solver."""
    return types.SimpleNamespace(
        _world=types.SimpleNamespace(cameras={}),
        _model=types.SimpleNamespace(body_label=("ground", "ball")),
        _lock=threading.RLock(),
    )


def _newton_add_camera(stub: types.SimpleNamespace, name: Any) -> dict[str, Any]:
    return NewtonSimEngine.add_camera(
        cast(NewtonSimEngine, stub), name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0]
    )


class TestNewtonAddCamera:
    @pytest.mark.parametrize("name", UNUSABLE)
    def test_an_unusable_name_is_refused_before_the_solver(self, name: str) -> None:
        stub = _newton_stub()
        result = _newton_add_camera(stub, name)
        assert result["status"] == "error", (name, result)
        assert "bare tokens" in result["content"][0]["text"]
        assert stub._world.cameras == {}

    @pytest.mark.parametrize("name", UNUSABLE)
    def test_the_refusal_is_the_same_sentence_as_mujocos(self, name: str) -> None:
        """One owner, one message: a caller moving between backends reads one rule."""
        result = _newton_add_camera(_newton_stub(), name)
        assert result["content"][0]["text"] == camera_name_error(
            "add_camera", "name", name, routes_free_camera_tokens=True
        )

    @pytest.mark.parametrize("name", USABLE)
    def test_a_usable_name_gets_past_the_guard(self, name: str) -> None:
        """Not a blanket refusal: a usable name reaches the solver path, which is what fails on the stub."""
        result = _newton_add_camera(_newton_stub(), name)
        text = result["content"][0]["text"]
        assert "bare tokens" not in text, (name, result)


# --------------------------------------------------------------------------- #
# Isaac                                                                       #
# --------------------------------------------------------------------------- #


class _FakeCameraHandle:
    """Stand-in for the Isaac ``Camera`` sensor handle."""


def _isaac_engine():
    from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation

    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig()
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._cameras = {}
    engine._prim_registry = []
    engine._cam_out_size = {}
    engine._camera_warmup_steps = 0
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()

    def _create_camera_prim(**kwargs: Any) -> tuple[Any, float]:
        return _FakeCameraHandle(), 24.0

    engine._create_camera_prim = _create_camera_prim  # type: ignore[method-assign]
    return engine


class TestIsaacAddCamera:
    """Isaac does not route the free-camera tokens, and this rule is not gated on that posture."""

    @pytest.mark.parametrize("name", UNUSABLE)
    def test_an_unusable_name_is_refused_before_the_stage(self, name: str) -> None:
        engine = _isaac_engine()
        result = engine.add_camera(name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "error", (name, result)
        assert "bare tokens" in result["content"][0]["text"]
        assert engine._cameras == {}

    @pytest.mark.parametrize("name", USABLE)
    def test_a_usable_name_still_registers(self, name: str) -> None:
        engine = _isaac_engine()
        result = engine.add_camera(name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "success", (name, result)
        assert name in engine._cameras

    def test_the_documented_signature_default_still_works(self) -> None:
        engine = _isaac_engine()
        assert engine.add_camera(position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])["status"] == "success"
        assert "default" in engine._cameras
