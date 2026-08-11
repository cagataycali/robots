"""``LiberoAdapter(cameras=...)`` refuses a per-camera config it cannot install.

Each value of the ``cameras`` mapping is documented as the keyword arguments
forwarded to ``Simulation.add_camera``, so a key that method does not declare
cannot be honored on any call - the splat raises :class:`TypeError` before the
camera is created. The install loop is deliberately tolerant of a sim *failing*
to add a camera (one flaky camera must not kill a whole eval) and that tolerance
used to cover this case too, which made a one-character typo silently equivalent
to omitting the camera: the LIBERO policy's required ``image`` / ``wrist_image``
view never entered the world, and every subsequent inference failed for a reason
unrelated to the policy under test.

The two halves this pins:

* the caller's mistake is refused, on the add path *and* on the skip path (the
  skip path forwards the same mapping to ``_publish_camera_dims_to_world``,
  which reads ``width`` / ``height`` by name, so a misspelled dimension used to
  publish the 256x256 fallback for a model-side camera instead);
* the resilience contract is untouched - an ``add_camera`` that raises, or that
  reports the camera already exists, is still best-effort.

The accepted key set is read from the sim's own ``add_camera`` rather than
hard-coded: MuJoCo and Newton declare ``parent_body`` and Isaac does not, so
hard-coding either would refuse a key a backend honors or accept one it cannot.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import pytest

from strands_robots.benchmarks.libero.adapter import camera_config_error
from tests.simulation.mujoco._gl_probe import requires_gl

from .test_libero_camera_install_resilience import (
    PICK_CUBE_BDDL,
    _adapter,
    _FakeSim,
)

# A good config for one camera, in the shape the adapter documents.
GOOD: dict[str, Any] = {
    "position": [1.0, 0.0, 1.5],
    "target": [0.0, 0.0, 0.85],
    "fov": 60.0,
    "width": 320,
    "height": 320,
}


def _mujoco_shaped_add_camera(
    name: str,
    position: list[float] | None = None,
    target: list[float] | None = None,
    fov: float = 60.0,
    width: int = 640,
    height: int = 480,
    parent_body: str | None = None,
) -> dict[str, Any]:
    """The MuJoCo / Newton ``add_camera`` signature, which declares ``parent_body``."""
    return {"status": "success"}


def _isaac_shaped_add_camera(
    name: str,
    position: list[float] | None = None,
    target: list[float] | None = None,
    width: int = 640,
    height: int = 480,
    fov: float = 60.0,
) -> dict[str, Any]:
    """The Isaac ``add_camera`` signature, which does NOT declare ``parent_body``."""
    return {"status": "success"}


def _recording_add_camera(calls: list[tuple[str, dict[str, Any]]]) -> Any:
    """A MuJoCo-shaped ``add_camera`` that records what it was called with."""

    def _add_camera(
        name: str,
        position: list[float] | None = None,
        target: list[float] | None = None,
        fov: float = 60.0,
        width: int = 640,
        height: int = 480,
        parent_body: str | None = None,
    ) -> dict[str, Any]:
        calls.append((name, {"width": width, "height": height, "fov": fov}))
        return {"status": "success"}

    return _add_camera


class TestTheAcceptedKeySetComesFromTheSim:
    """The domain is the backend's, read from the bound method."""

    def test_a_key_the_signature_declares_is_accepted(self):
        assert camera_config_error(_mujoco_shaped_add_camera, "image", GOOD) is None

    def test_parent_body_is_accepted_where_the_backend_declares_it(self):
        cfg = {**GOOD, "parent_body": "robot0_right_hand"}
        assert camera_config_error(_mujoco_shaped_add_camera, "wrist_image", cfg) is None

    def test_parent_body_is_refused_where_the_backend_does_not_declare_it(self):
        cfg = {**GOOD, "parent_body": "robot0_right_hand"}
        msg = camera_config_error(_isaac_shaped_add_camera, "wrist_image", cfg)
        assert msg is not None
        assert "'parent_body'" in msg
        # Hard-coding one backend's set would have accepted this.
        assert "parent_body" not in msg.split("Accepted keys:")[1]

    def test_a_signature_taking_var_keyword_accepts_any_key(self):
        def _tolerant(name: str, **kwargs: Any) -> None:
            return None

        assert camera_config_error(_tolerant, "image", {**GOOD, "anything": 1}) is None

    def test_a_non_introspectable_callable_defers_to_the_call(self):
        """A C-implemented callable whose signature CPython cannot report must
        not be turned into a caller error - there is nothing to check against,
        so the call itself stays the judge."""
        opaque = min  # stands in for a callable with no reportable signature
        with pytest.raises(ValueError, match="no signature found"):
            inspect.signature(opaque)  # premise: the branch under test is reached

        assert camera_config_error(opaque, "image", {"unknowable": 1}) is None

    def test_an_empty_config_is_accepted(self):
        assert camera_config_error(_mujoco_shaped_add_camera, "image", {}) is None


class TestTheMessageNamesEveryUnusableKey:
    """A refusal has to be actionable without reading the backend's source."""

    def test_a_typo_is_named_with_a_suggestion(self):
        msg = camera_config_error(_mujoco_shaped_add_camera, "image", {**GOOD, "heigth": 320})
        assert msg is not None
        assert "cameras['image']" in msg
        assert "'heigth'" in msg
        assert "did you mean 'height'?" in msg

    def test_an_unrelated_key_is_named_without_a_wrong_suggestion(self):
        msg = camera_config_error(_mujoco_shaped_add_camera, "image", {"resolution": [320, 320]})
        assert msg is not None
        assert "'resolution'" in msg
        assert "did you mean" not in msg

    def test_every_unusable_key_is_named(self):
        msg = camera_config_error(_mujoco_shaped_add_camera, "image", {"positon": [0.0, 0.0, 1.0], "fovy": 60.0})
        assert msg is not None
        assert "'fovy'" in msg
        assert "'positon'" in msg

    def test_the_accepted_keys_are_listed(self):
        msg = camera_config_error(_mujoco_shaped_add_camera, "image", {"nope": 1})
        assert msg is not None
        listed = msg.split("Accepted keys:")[1]
        for key in ("position", "target", "fov", "width", "height", "parent_body"):
            assert key in listed
        # ``name`` is supplied by the install, so offering it would be wrong.
        assert "name" not in listed

    def test_the_reserved_name_key_is_refused_with_its_own_reason(self):
        msg = camera_config_error(_mujoco_shaped_add_camera, "image", {**GOOD, "name": "other"})
        assert msg is not None
        assert "'name' must not be set" in msg

    def test_the_reserved_name_key_is_refused_even_by_a_tolerant_signature(self):
        def _tolerant(name: str, **kwargs: Any) -> None:
            return None

        msg = camera_config_error(_tolerant, "image", {"name": "other"})
        assert msg is not None
        assert "'name' must not be set" in msg


class TestInstallRefusesAConfigItCannotHonor:
    """The install loop raises instead of leaving the camera silently absent."""

    def test_an_unusable_key_raises_before_any_camera_is_added(self):
        adapter = _adapter()
        adapter._cameras = {"image": {**GOOD, "heigth": 320}}
        calls: list[tuple[str, dict[str, Any]]] = []
        sim = _FakeSim(_recording_add_camera(calls))

        with pytest.raises(ValueError, match="does not accept 'heigth'"):
            adapter._install_libero_cameras(sim)  # type: ignore[arg-type]

        # The refusal precedes the side effect: nothing was half-installed.
        assert calls == []
        assert sim._world.cameras == {}

    def test_the_skip_path_refuses_too_instead_of_publishing_default_dims(self):
        """A camera already in the model took the skip branch, which forwards the
        same mapping to ``_publish_camera_dims_to_world``. That reads ``width`` /
        ``height`` by name, so a misspelled dimension used to publish the
        256x256 fallback under a successful install."""
        adapter = _adapter()
        adapter._cameras = {"image": {**GOOD, "heigth": 320}}
        calls: list[tuple[str, dict[str, Any]]] = []
        sim = _FakeSim(_recording_add_camera(calls))
        # Mimic a scene MJCF that already declared the camera.
        sim._world.cameras["image"] = object()

        with pytest.raises(ValueError, match="does not accept 'heigth'"):
            adapter._install_libero_cameras(sim)  # type: ignore[arg-type]

        assert calls == []

    def test_a_usable_config_still_installs(self):
        adapter = _adapter()
        adapter._cameras = {"image": dict(GOOD)}
        calls: list[tuple[str, dict[str, Any]]] = []
        sim = _FakeSim(_recording_add_camera(calls))

        adapter._install_libero_cameras(sim)  # type: ignore[arg-type]

        assert [name for name, _ in calls] == ["image"]
        assert calls[0][1]["width"] == 320
        assert calls[0][1]["height"] == 320


class TestTheResilienceContractIsUntouched:
    """A sim *failing* to add a camera is still best-effort - only a config the
    backend can never accept is refused."""

    def test_a_raising_add_camera_is_still_swallowed(self, caplog):
        adapter = _adapter()
        adapter._cameras = {"image": dict(GOOD), "wrist_image": dict(GOOD)}
        attempted: list[str] = []

        def _raising(
            name: str,
            position: list[float] | None = None,
            target: list[float] | None = None,
            fov: float = 60.0,
            width: int = 640,
            height: int = 480,
            parent_body: str | None = None,
        ) -> None:
            attempted.append(name)
            raise RuntimeError(f"boom installing {name}")

        sim = _FakeSim(_raising)
        with caplog.at_level(logging.WARNING, logger="strands_robots.benchmarks.libero.adapter"):
            assert adapter._install_libero_cameras(sim) is None  # type: ignore[arg-type]

        # Every camera was still attempted: a transient failure is not a caller error.
        assert set(attempted) == {"image", "wrist_image"}

    def test_an_already_exists_error_still_publishes_the_configured_dims(self):
        adapter = _adapter()
        adapter._cameras = {"image": dict(GOOD)}

        def _already_exists(
            name: str,
            position: list[float] | None = None,
            target: list[float] | None = None,
            fov: float = 60.0,
            width: int = 640,
            height: int = 480,
            parent_body: str | None = None,
        ) -> dict[str, Any]:
            return {"status": "error", "content": [{"text": f"camera {name!r} already exists"}]}

        sim = _FakeSim(_already_exists)
        adapter._install_libero_cameras(sim)  # type: ignore[arg-type]

        published = sim._world.cameras["image"]
        assert (published.width, published.height) == (320, 320)


class TestOnTheRealMuJoCoBackend:
    """End to end against the backend whose signature the domain is read from."""

    @staticmethod
    def _engine(tool_name: str):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        sim = MuJoCoSimEngine(tool_name=tool_name, mesh=False)
        sim.create_world()
        sim.add_robot(name="panda", data_config="panda")
        return sim

    def test_the_accepted_set_matches_the_real_signature(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        declared = {p for p in inspect.signature(MuJoCoSimEngine.add_camera).parameters if p not in ("self", "name")}
        msg = camera_config_error(MuJoCoSimEngine.add_camera, "image", {"nope": 1})
        assert msg is not None
        listed = msg.split("Accepted keys:")[1]
        assert declared and all(key in listed for key in declared)

    def test_a_usable_config_installs(self):
        sim = self._engine("libero_cam_domain_good")
        try:
            adapter = _adapter()
            adapter._cameras = {"image": dict(GOOD)}
            adapter._install_libero_cameras(sim)

            assert "image" in sim._world.cameras
            entry = sim._world.cameras["image"]
            assert (entry.width, entry.height) == (320, 320)
        finally:
            sim.cleanup()

    @requires_gl
    def test_a_usable_config_renders(self):
        """The installed camera is renderable, which needs a host GL context.

        Split from the install case so the resolution the domain governs is
        still pinned on a headless host without EGL/OSMesa, where ``render``
        reports an error for a reason unrelated to the camera config.
        """
        sim = self._engine("libero_cam_domain_render")
        try:
            adapter = _adapter()
            adapter._cameras = {"image": dict(GOOD)}
            adapter._install_libero_cameras(sim)

            assert sim.render(camera_name="image", width=64, height=64)["status"] == "success"
        finally:
            sim.cleanup()

    def test_a_typo_is_refused_and_the_camera_never_enters_the_world(self):
        sim = self._engine("libero_cam_domain_typo")
        try:
            adapter = _adapter()
            adapter._cameras = {"image": {**GOOD, "heigth": 320}}

            with pytest.raises(ValueError, match="did you mean 'height'"):
                adapter._install_libero_cameras(sim)

            assert "image" not in sim._world.cameras
            # The view the policy reads is simply not there.
            assert sim.render(camera_name="image", width=64, height=64)["status"] == "error"
        finally:
            sim.cleanup()


def test_the_bddl_fixture_is_the_shared_one():
    """Premise: the reused fixture really parses, so an adapter exists to test."""
    assert "grasped cube_1" in PICK_CUBE_BDDL
    assert set(_adapter()._cameras) == {"image", "wrist_image"}
