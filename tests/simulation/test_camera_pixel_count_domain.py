# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: every backend's camera pixel dimensions share one floor.

``width`` / ``height`` reach two surfaces on every simulation backend - the
``add_camera`` that fixes a camera's resolution, and the render family
(``render`` / ``get_frame`` / ``get_camera_params``) that can override it per
call. MuJoCo validated both through ``_validate_render_dims``; Newton and Isaac
validated neither, and coerced with a bare ``int(...)`` that refuses nothing
useful. Coercion is validation only when the coercion rejects.

Measured on both backends before the fix:

* **An uncatchable exception through the tool-result contract.** ``width="big"``
  raised ``ValueError: invalid literal for int() with base 10: 'big'`` out of
  ``add_camera``, ``None`` a ``TypeError``, ``nan`` a ``ValueError`` and ``inf``
  an ``OverflowError``. On Isaac the ``int()`` sat *outside* the method's try
  block, so none of them could even be converted into an error envelope.
* **A camera that cannot produce a frame, reported as success.**
  ``add_camera(width=0, height=-4)`` returned ``status="success"`` and stored the
  values, deferring the failure to the first render - far from the call that
  caused it. On Isaac a stored negative width made every later render of that
  camera fail (``ValueError: negative dimensions are not allowed`` out of the
  blank-frame ``np.zeros``), so one bad configuration call disabled the camera
  for the rest of the session.
* **Truthiness where membership was meant.** Isaac's
  ``int(width or self._config.camera_width)`` and Newton's
  ``int(width or self.default_width)`` read a falsy ``0`` as *omitted*, so the
  caller asked for an impossible resolution and was handed the default while
  being told it was what they requested - the harder failure to notice of the
  two, and the same shape as the ``position or <default>`` bug
  ``coerce_pose_vector``'s docstring documents.
* **Silent truncation, with the success text disagreeing with the registry.**
  ``width=2.7`` stored 2 and ``True`` stored 1, while Newton's success message
  echoed the caller's raw value (``"Camera 'cam' added (2.7x480, ...)"`` /
  ``"(Truex480, ...)"``) - reporting a resolution that was never registered.

The fix routes all four surfaces through
:func:`~strands_robots.utils.positive_count_error`, the domain MuJoCo's floor
already implements: a true ``int`` (``bool`` refused - it is an ``int`` subclass
whose ``True`` would act as a silent 1) that is ``>= 1``. A pixel dimension is
consumed directly as an array / framebuffer dimension, where an integral float
raises ``TypeError`` rather than being coerced, which is why it belongs to that
domain rather than to the looser
:func:`~strands_robots.utils.positive_whole_number_error` used for frame rates.
The *upper* bound stays backend-specific: MuJoCo caps at its offscreen
framebuffer, and the ray-traced backends have no such buffer, so
``TestSameVerdictAcrossBackends`` pins agreement on the floor only.

None of this needs ``newton`` / ``warp`` / Isaac Sim or a GPU: the validation
runs before either backend touches its solver or its stage, so the newton-free
stub and the ``__new__`` Isaac skeleton the sibling ``add_camera`` tests
already establish exercise it in every environment.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import _MIN_RENDER_PX
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import positive_count_error

from .isaac.test_add_camera_numeric_validation import _engine as _isaac_engine
from .newton.test_add_camera_numeric_validation import _engine_stub

#: Pixel dimensions no camera can be built or rendered at. Each was accepted or
#: raised past the tool-result contract on both backends before the fix.
_BAD_DIMS: tuple[Any, ...] = (
    0,
    -4,
    -1,
    2.7,
    640.0,
    True,
    False,
    "big",
    "640",
    float("nan"),
    float("inf"),
    float("-inf"),
    [640],
)

#: Dimensions that must keep working. ``5000`` is included deliberately: it is
#: above MuJoCo's framebuffer cap but perfectly renderable by a ray tracer, so
#: it pins that the shared rule is a floor and not a ceiling.
_GOOD_DIMS: tuple[int, ...] = (1, 16, 320, 640, 5000)


def _newton_stub() -> types.SimpleNamespace:
    """The sibling test's ``add_camera`` stub, plus what the render family reads.

    ``_resolve_camera_view`` additionally reads ``self.default_width`` /
    ``self.default_height`` (the built-in view) and calls
    ``_resolve_camera_pose`` (which, for a world-fixed camera, returns the
    stored pose without touching the solver). Both it and ``list_cameras`` are
    bound as real methods rather than replaced, so ``render`` exercises the
    production resolver instead of a stand-in for it.
    """
    stub = _engine_stub()
    stub.default_width = 640
    stub.default_height = 480
    for method in ("_resolve_camera_pose", "_resolve_camera_view", "list_cameras"):
        setattr(stub, method, types.MethodType(getattr(NewtonSimEngine, method), stub))
    return stub


def _newton_add_camera(stub: types.SimpleNamespace, **kwargs: Any) -> dict[str, Any]:
    """Call ``add_camera`` with arguments deliberately outside its annotations.

    Every bad dimension below is a value a caller must never pass, because the
    contract under test is that it is refused at runtime - in a structured
    result - rather than by a type checker that an agent dispatching this tool
    never runs. Funnelling the calls through one ``**kwargs: Any`` boundary
    states that once, the shape both sibling ``add_camera`` tests use, instead
    of scattering per-call suppressions.
    """
    return NewtonSimEngine.add_camera(stub, **kwargs)  # type: ignore[arg-type]


def _newton_resolve(stub: types.SimpleNamespace, camera_name: str | None, **kwargs: Any) -> tuple:
    """``_resolve_camera_view``, the funnel Newton's three render surfaces share."""
    return NewtonSimEngine._resolve_camera_view(stub, camera_name, kwargs.get("width"), kwargs.get("height"))  # type: ignore[arg-type]


def _verdict(call: Callable[[], Any]) -> str:
    """``"refused"`` / ``"accepted"``; an exception the API leaked is its own verdict.

    A leaked exception is reported rather than swallowed: it is a contract
    violation in its own right, and folding it into a refusal would let the
    parity tests below pass against the pre-fix code that raised ``ValueError``
    out of ``int()``. Every type that code leaked - ``ValueError``,
    ``TypeError``, ``OverflowError`` - is an ``Exception``, which is where the
    boundary sits.

    It deliberately does not sit at ``BaseException``. A ``KeyboardInterrupt``
    or a pytest outcome (``Skipped`` / ``Failed``, both of which derive from
    ``BaseException`` without deriving from ``Exception``) is not something the
    API under test leaked - it is the operator's or the framework's control flow
    passing through. Absorbing those would turn an interrupted run into a
    verdict string, and an ``importorskip`` in one of the helpers into a
    confusing failure instead of a skip, so they propagate.
    """
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 - an exception the API leaked is the finding
        return f"raised {type(exc).__name__}"
    return "refused" if result["status"] == "error" else "accepted"


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheSharedDomain:
    """What ``positive_count_error`` accepts is what a pixel dimension can be."""

    @pytest.mark.parametrize("bad", _BAD_DIMS)
    def test_every_probe_value_is_outside_the_domain(self, bad):
        """Pin the premise so a weakened guard cannot make this file vacuous."""
        assert positive_count_error(bad, "width", "render") is not None

    @pytest.mark.parametrize("good", _GOOD_DIMS)
    def test_every_good_value_is_inside_the_domain(self, good):
        assert positive_count_error(good, "width", "render") is None

    def test_an_integral_float_is_refused_because_it_cannot_be_used_as_a_dimension(self):
        """``640.0`` is outside the domain, and this is why.

        The looser :func:`positive_whole_number_error` accepts it - correct for a
        frame rate read out of a config float. A pixel dimension is handed
        straight to an array constructor, which refuses it, so the two knobs
        genuinely need different domains rather than one shared spelling.
        """
        with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
            np.zeros((480, 640.0, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Newton                                                                      #
# --------------------------------------------------------------------------- #
class TestNewtonAddCamera:
    @pytest.mark.parametrize("bad", _BAD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_unusable_dimension_is_refused(self, param, bad):
        """Pre-fix these were stored verbatim, truncated, or raised out of the call."""
        stub = _newton_stub()
        result = _newton_add_camera(stub, name="cam", **{param: bad})
        assert result["status"] == "error", result
        assert result["content"][0]["text"] == f"add_camera: {param} must be a positive integer, got {bad!r}."
        assert stub._world.cameras == {}

    @pytest.mark.parametrize("good", _GOOD_DIMS)
    def test_usable_dimension_is_stored_and_reported_as_given(self, good):
        """The registry and the success text must agree on the resolution.

        Pre-fix the message interpolated the caller's raw value while the
        registry held ``int(...)`` of it, so ``width=2.7`` announced a ``2.7``
        pixel-wide camera that was actually 2 wide.
        """
        stub = _newton_stub()
        result = _newton_add_camera(stub, name="cam", width=good, height=good)
        assert result["status"] == "success", result
        cam = stub._world.cameras["cam"]
        assert (cam.width, cam.height) == (good, good)
        assert f"{good}x{good}" in result["content"][0]["text"]

    def test_defaults_are_unchanged(self):
        stub = _newton_stub()
        assert _newton_add_camera(stub, name="cam")["status"] == "success"
        cam = stub._world.cameras["cam"]
        assert (cam.width, cam.height) == (640, 480)

    def test_a_refused_dimension_does_not_consume_the_camera_name(self):
        """A rejected configuration leaves the registry untouched.

        The name must still be free afterwards - otherwise the caller has to
        ``remove_camera`` a camera that was never added.
        """
        stub = _newton_stub()
        assert _newton_add_camera(stub, name="wrist", width=0)["status"] == "error"
        assert _newton_add_camera(stub, name="wrist", width=320)["status"] == "success"
        assert stub._world.cameras["wrist"].width == 320


class TestNewtonRenderFamily:
    """``_resolve_camera_view`` is the funnel ``render`` / ``get_frame`` /
    ``get_camera_params`` share, so one guard covers all three."""

    @pytest.mark.parametrize("bad", _BAD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    @pytest.mark.parametrize("camera", ["default", "cam"])
    def test_unusable_override_is_refused(self, camera, param, bad):
        stub = _newton_stub()
        assert _newton_add_camera(stub, name="cam", width=320, height=240)["status"] == "success"
        with pytest.raises(ValueError, match=f"render: {param} must be a positive integer"):
            _newton_resolve(stub, camera, **{param: bad})

    def test_zero_is_refused_rather_than_read_as_omitted(self):
        """Membership, not truthiness.

        Pre-fix ``int(width or self.default_width)`` resolved ``width=0`` to the
        engine default and rendered at 640 wide, reporting that as the size the
        caller requested. ``0`` is a request for an impossible resolution, not an
        omission - and ``bool(0) is False`` is the whole reason it was mistaken
        for one.
        """
        assert not bool(0)
        stub = _newton_stub()
        with pytest.raises(ValueError, match="width must be a positive integer, got 0"):
            _newton_resolve(stub, "default", width=0)

    @pytest.mark.parametrize("camera", ["default", "cam"])
    def test_none_still_means_take_the_configured_size(self, camera):
        stub = _newton_stub()
        assert _newton_add_camera(stub, name="cam", width=320, height=240)["status"] == "success"
        _eye, _target, _fov, w, h = _newton_resolve(stub, camera)
        assert (w, h) == ((640, 480) if camera == "default" else (320, 240))

    @pytest.mark.parametrize("good", _GOOD_DIMS)
    def test_usable_override_is_honored_exactly(self, good):
        stub = _newton_stub()
        assert _newton_add_camera(stub, name="cam", width=320, height=240)["status"] == "success"
        _eye, _target, _fov, w, h = _newton_resolve(stub, "cam", width=good, height=good)
        assert (w, h) == (good, good)

    def test_render_reports_the_refusal_as_a_structured_error(self):
        """``render`` owes its caller an envelope, not an exception.

        It already converts a ``ValueError`` from the shared resolver into one,
        which is why the guard raises there rather than being duplicated per
        entry point.
        """
        stub = _newton_stub()
        result = NewtonSimEngine.render(stub, camera_name="default", width=0)  # type: ignore[arg-type]
        assert result["status"] == "error", result
        assert "width must be a positive integer" in result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# Isaac                                                                       #
# --------------------------------------------------------------------------- #
class TestIsaacAddCamera:
    @pytest.mark.parametrize("bad", _BAD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_unusable_dimension_is_refused_before_the_stage_is_touched(self, param, bad):
        engine = _isaac_engine()
        result = engine.add_camera(name="cam", **{param: bad})
        assert result["status"] == "error", result
        assert result["content"][0]["text"] == f"add_camera: {param} must be a positive integer, got {bad!r}."
        assert engine._cameras == {}
        # The prim recorder never fired: nothing was created on the stage, so
        # there is no partial camera to clean up.
        assert engine.prim_calls == []

    @pytest.mark.parametrize("good", _GOOD_DIMS)
    def test_usable_dimension_is_honored(self, good):
        engine = _isaac_engine()
        result = engine.add_camera(name="cam", width=good, height=good)
        assert result["status"] == "success", result
        assert (engine._cameras["cam"].width, engine._cameras["cam"].height) == (good, good)

    def test_zero_is_refused_rather_than_read_as_the_config_default(self):
        """Pre-fix ``int(width or camera_width)`` reported ``640x480`` for ``width=0``.

        The caller asked for a resolution no camera can have and was told the
        config default was what they requested - a success naming a size that
        was never requested is harder to notice than a stored ``0``.
        """
        engine = _isaac_engine()
        result = engine.add_camera(name="cam", width=0)
        assert result["status"] == "error", result
        assert engine._cameras == {}

    def test_omitted_dimensions_still_take_the_config_defaults(self):
        engine = _isaac_engine()
        assert engine.add_camera(name="cam")["status"] == "success"
        cam = engine._cameras["cam"]
        assert (cam.width, cam.height) == (engine._config.camera_width, engine._config.camera_height)

    def test_the_dlss_upscale_would_have_amplified_a_negative_width(self):
        """Pin why a non-positive width was worse here than a mis-sized frame.

        ``add_camera`` raises the native render size to ``_MIN_RENDER_PX`` when
        the requested width is below it, preserving the aspect ratio via
        ``scale = _MIN_RENDER_PX / w``. For ``w = -4`` that scale is negative, so
        the height was multiplied out to ``-76800`` and stored - and every later
        render of that camera then failed on the negative dimension. Refusing
        ``w`` at the entry point is what keeps this arithmetic unreachable.
        """
        w = -4
        scale = _MIN_RENDER_PX / float(w)
        assert int(round(480 * scale)) == -76800
        with pytest.raises(ValueError, match="negative dimensions"):
            np.zeros((int(round(480 * scale)), _MIN_RENDER_PX, 3), dtype=np.uint8)


class TestIsaacRenderFamily:
    @pytest.mark.parametrize("bad", _BAD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_render_reports_an_unusable_override_as_an_error(self, param, bad):
        """``_render_frame`` owes its caller ``(rgb, depth, meta)``, never an exception.

        Pre-fix a negative dimension reached ``np.zeros`` as ``ValueError:
        negative dimensions are not allowed`` and a fractional or non-numeric one
        as ``TypeError``, both escaping that contract.
        """
        engine = _isaac_engine()
        rgb, depth, meta = engine._render_frame("default", **{param: bad})
        assert rgb is None and depth is None
        assert meta["error"] == f"render: {param} must be a positive integer, got {bad!r}."

    def test_render_envelope_carries_the_refusal(self):
        engine = _isaac_engine()
        result = engine.render(camera_name="default", width=-4)
        assert result["status"] == "error", result
        assert "width must be a positive integer" in result["content"][0]["text"]

    @pytest.mark.parametrize("good", (16, 320, 640))
    def test_usable_override_still_renders(self, good):
        engine = _isaac_engine()
        rgb, _depth, meta = engine._render_frame("default", good, good)
        assert rgb is not None and rgb.shape == (good, good, 3), meta

    def test_none_still_means_take_the_config_default(self):
        engine = _isaac_engine()
        rgb, _depth, _meta = engine._render_frame("default")
        assert rgb is not None
        assert rgb.shape == (engine._config.camera_height, engine._config.camera_width, 3)


# --------------------------------------------------------------------------- #
# Cross-backend parity                                                        #
# --------------------------------------------------------------------------- #
class TestSameVerdictAcrossBackends:
    """A pixel dimension one backend refuses must be refused by all of them.

    The headline inconsistency this closes was exactly that divergence: MuJoCo
    refused every value below at both surfaces, Newton and Isaac accepted most
    of them and raised on the rest. Parity is pinned here rather than by sharing
    the message text, because the *upper* bound is legitimately
    backend-specific - MuJoCo has an offscreen framebuffer to overflow and the
    ray tracers do not - so only the floor is common.
    """

    @pytest.fixture
    def mj_sim(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        s = Simulation(tool_name="test_camera_pixel_count_parity_sim", mesh=False)
        s.create_world(gravity=[0, 0, -9.81])
        yield s
        s.cleanup()

    @pytest.mark.parametrize("bad", _BAD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_add_camera_agrees_on_the_floor(self, mj_sim, param, bad):
        mujoco_verdict = _verdict(lambda: mj_sim.add_camera(name="parity_cam", **{param: bad}))
        newton_verdict = _verdict(lambda: _newton_add_camera(_newton_stub(), name="parity_cam", **{param: bad}))
        isaac_verdict = _verdict(lambda: _isaac_engine().add_camera(name="parity_cam", **{param: bad}))
        assert mujoco_verdict == newton_verdict == isaac_verdict == "refused", (
            f"{param}={bad!r}: mujoco={mujoco_verdict}, newton={newton_verdict}, isaac={isaac_verdict}"
        )

    @pytest.mark.parametrize("bad", _BAD_DIMS)
    def test_the_render_family_agrees_on_the_floor(self, mj_sim, bad):
        mujoco_verdict = _verdict(lambda: mj_sim.render(camera_name="default", width=bad, height=480))
        isaac_verdict = _verdict(lambda: _isaac_engine().render(camera_name="default", width=bad))
        newton_verdict = _verdict(
            lambda: NewtonSimEngine.render(_newton_stub(), camera_name="default", width=bad)  # type: ignore[arg-type]
        )
        assert mujoco_verdict == newton_verdict == isaac_verdict == "refused", (
            f"width={bad!r}: mujoco={mujoco_verdict}, newton={newton_verdict}, isaac={isaac_verdict}"
        )

    @pytest.mark.parametrize("good", (16, 320, 640))
    def test_a_usable_dimension_is_accepted_everywhere(self, mj_sim, good):
        """The guard is a floor, not a tightening: ordinary sizes still work."""
        assert _verdict(lambda: mj_sim.add_camera(name=f"ok_{good}", width=good, height=good)) == "accepted"
        assert _verdict(lambda: _newton_add_camera(_newton_stub(), name="ok", width=good, height=good)) == "accepted"
        assert _verdict(lambda: _isaac_engine().add_camera(name="ok", width=good, height=good)) == "accepted"


class TestTheVerdictClassifier:
    """The helper the parity tests share must not absorb control flow.

    :func:`_verdict` turns one call into ``"refused"`` / ``"accepted"`` /
    ``"raised X"`` so a divergence between three backends is reported in a
    single assertion message. That reporting is sound only for exceptions the
    API leaked, so the boundary is pinned here in both directions: a classifier
    that also caught pytest's own outcomes would report a skipped optional
    dependency as a failure, and an interrupted run as a verdict.
    """

    def test_an_envelope_is_classified_by_its_status(self):
        """The two ordinary outcomes: a structured refusal and an acceptance."""
        assert _verdict(lambda: {"status": "error"}) == "refused"
        assert _verdict(lambda: {"status": "success"}) == "accepted"

    @pytest.mark.parametrize("leaked", [ValueError, TypeError, OverflowError])
    def test_an_exception_the_api_leaked_becomes_a_verdict(self, leaked):
        """Each type the pre-fix backends leaked out of ``int()`` is still reported.

        Reporting rather than propagating is what stops the parity tests passing
        against code that raises instead of refusing, so it has to survive the
        narrowing of the caught set.
        """

        def leak() -> dict[str, Any]:
            raise leaked("leaked past the tool-result contract")

        assert _verdict(leak) == f"raised {leaked.__name__}"

    @pytest.mark.parametrize(
        "control_flow",
        [pytest.skip.Exception, pytest.fail.Exception, KeyboardInterrupt, SystemExit],
    )
    def test_control_flow_is_not_absorbed(self, control_flow):
        """A pytest outcome or an operator interrupt passes straight through.

        All four derive from ``BaseException`` without deriving from
        ``Exception``, and none of them is the API's to leak. Absorbing them
        would turn a skipped dependency into a failure and a ``Ctrl-C`` into the
        string ``"raised KeyboardInterrupt"``, compared against the other
        backends' verdicts as though it were an answer.
        """

        def interrupted() -> dict[str, Any]:
            raise control_flow("not the API's to leak")

        with pytest.raises(control_flow):
            _verdict(interrupted)
