# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: every backend's camera pixel dimensions share one floor.

``width`` / ``height`` reach two surfaces on every simulation backend - the
``add_camera`` that fixes a camera's resolution, and the render family
(``render`` / ``get_frame`` / ``get_camera_params``) that can override it per
call - and a third on Newton alone, ``open_viewer``, which sizes the ``"gl"``
window. MuJoCo validated the first two through ``_validate_render_dims``;
Newton and Isaac validated neither, and coerced with a bare ``int(...)`` that
refuses nothing useful. Coercion is validation only when the coercion rejects.

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

* **One quantity, two contracts, decided by which method was called.** Newton's
  ``open_viewer`` sized its ``"gl"`` window from the same "how many pixels"
  value and applied no domain at all, so 12 of the 13 dimensions below were
  refused by ``render`` and forwarded verbatim into ``ViewerGL`` on the same
  engine, in the same session. The consequence is the one the neighbouring
  ``port`` guard already states in its own comment: the engine holds a single
  viewer slot, so ``open_viewer("gl", width=0, height=0)`` returned
  ``status="success"``, built a zero-pixel window and filled the slot - after
  which the obvious recovery, calling ``open_viewer`` again with a usable size,
  was answered ``"Viewer already open (gl)."`` under ``status="success"`` and
  built nothing. The caller was left with the window they did not ask for and
  no way to replace it.

The fix routes all five surfaces through
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

import ast
import inspect
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import _MIN_RENDER_PX
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import positive_count_error

from .isaac.test_add_camera_numeric_validation import _engine as _isaac_engine
from .newton.test_add_camera_numeric_validation import _engine_stub
from .newton.test_viewer_port_domain import _viewer_stub

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
# Newton: the interactive viewer window                                       #
# --------------------------------------------------------------------------- #
def _newton_open_viewer(stub: Any, **kwargs: Any) -> dict[str, Any]:
    """``open_viewer`` with arguments deliberately outside its annotations.

    Same ``**kwargs: Any`` boundary as ``_newton_add_camera`` above, and for the
    same reason: the contract under test is what happens at runtime to a value
    an agent dispatching this tool can pass, not what a type checker it never
    runs would have said.
    """
    return NewtonSimEngine.open_viewer(stub, **kwargs)  # type: ignore[arg-type]


class TestNewtonViewerDimensions:
    """The ``"gl"`` window size is the same floor as a frame's.

    ``_viewer_stub(has_display=True)`` is the sibling port-domain file's stand-in
    ``self``: it records every viewer it constructs, so these tests assert a
    refusal built *nothing* rather than only that it reported an error. The guard
    precedes the lock and every viewer construction, so none of this needs
    ``newton`` / ``warp`` or a display.
    """

    @pytest.mark.parametrize("bad", _BAD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_unusable_dimension_is_refused(self, param, bad):
        stub = _viewer_stub(has_display=True)
        result = _newton_open_viewer(stub, viewer="gl", **{param: bad})
        assert result["status"] == "error", (param, bad, result)
        assert param in result["content"][0]["text"]

    @pytest.mark.parametrize("bad", _BAD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_a_refused_dimension_constructs_no_viewer(self, param, bad):
        """Nothing is built, so nothing occupies the single viewer slot."""
        stub = _viewer_stub(has_display=True)
        _newton_open_viewer(stub, viewer="gl", **{param: bad})
        assert stub.built == []
        assert stub._viewer is None

    @pytest.mark.parametrize("param", ["width", "height"])
    def test_the_retry_after_a_refusal_still_opens(self, param):
        """The recovery a forwarded value denied: refuse, then open normally.

        This is the property the defect removed. Pre-fix the refused call built a
        zero-pixel window and took the slot, so this second call returned
        ``"Viewer already open"`` - a success result for a window the caller
        never asked for, with no way to replace it.
        """
        stub = _viewer_stub(has_display=True)
        assert _newton_open_viewer(stub, viewer="gl", **{param: 0})["status"] == "error"
        retry = _newton_open_viewer(stub, viewer="gl", width=1280, height=720)
        assert retry["status"] == "success", retry
        assert stub.built == [("gl", {"width": 1280, "height": 720})]

    @pytest.mark.parametrize("good", _GOOD_DIMS)
    def test_a_usable_dimension_is_forwarded_exactly(self, good):
        """A floor, not a tightening - and the value reaches ``ViewerGL`` unchanged."""
        stub = _viewer_stub(has_display=True)
        result = _newton_open_viewer(stub, viewer="gl", width=good, height=good)
        assert result["status"] == "success", result
        assert stub.built == [("gl", {"width": good, "height": good})]

    def test_the_defaults_are_unchanged(self):
        stub = _viewer_stub(has_display=True)
        assert _newton_open_viewer(stub, viewer="gl")["status"] == "success"
        assert stub.built == [("gl", {"width": 1280, "height": 720})]

    def test_auto_resolving_to_gl_is_checked_too(self):
        """``"auto"`` becomes ``"gl"`` when a display is present, so it is the gl branch."""
        stub = _viewer_stub(has_display=True)
        assert _newton_open_viewer(stub, viewer="auto", width=0)["status"] == "error"
        assert stub.built == []

    def test_the_missing_display_is_still_reported_first(self):
        """A headless host's more fundamental problem keeps naming itself.

        Ordering matters: without a display no window of any size can open, so
        that refusal is the actionable one and the dimension guard must not
        shadow it.
        """
        stub = _viewer_stub(has_display=False)
        result = _newton_open_viewer(stub, viewer="gl", width=0)
        assert result["status"] == "error"
        assert "no display server" in result["content"][0]["text"]


class TestOnlyTheGlBranchReadsTheDimensions:
    """The domain is applied where the value is read, as ``port``'s already is.

    ``"viser"`` and ``"null"`` never pass ``width`` / ``height`` to a viewer, so
    refusing a dimension for them would be a false rejection - the same scoping
    the sibling ``port`` guard takes by checking only the viser branch.
    """

    @pytest.mark.parametrize("kind", ["viser", "null"])
    @pytest.mark.parametrize("bad", (0, -1, 2.7, "big", None))
    def test_a_branch_that_ignores_the_size_still_opens(self, kind, bad):
        stub = _viewer_stub(has_display=True)
        result = _newton_open_viewer(stub, viewer=kind, width=bad, height=bad)
        assert result["status"] == "success", (kind, bad, result)
        assert [built_kind for built_kind, _ in stub.built] == [kind]

    def test_the_port_domain_still_applies_to_viser(self):
        """Non-vacuity for the scoping above: viser is checked, on its own knob."""
        stub = _viewer_stub(has_display=True)
        result = _newton_open_viewer(stub, viewer="viser", port=0)
        assert result["status"] == "error", result
        assert "port" in result["content"][0]["text"]
        assert stub.built == []


def _render_dims_verdict(param: str, value: Any) -> str:
    """``"refused"`` / ``"accepted"`` for the funnel the render surfaces share.

    ``_resolve_camera_view`` is where ``render`` / ``get_frame`` /
    ``get_camera_params`` apply the domain, and it *raises* ``ValueError`` -
    ``render`` converts that into an envelope. It is used here rather than
    ``render`` itself because the newton-free stub cannot rasterize a frame, so a
    *usable* dimension would come back as a renderer error and read as a
    refusal. The funnel is the layer that answers the question this parity is
    about, and it is the same one ``TestNewtonRenderFamily`` above asserts on.
    """
    try:
        _newton_resolve(_newton_stub(), "default", **{param: value})
    except ValueError:
        return "refused"
    return "accepted"


class TestTheViewerAgreesWithTheRenderFamily:
    """One engine, one pixel quantity: the two surfaces cannot disagree.

    This is the assertion that fails pre-fix for every unusable row - the render
    family refused each of them and ``open_viewer`` accepted it, in the same
    session - and the reason the viewer belongs to this file rather than to a
    domain of its own.
    """

    @pytest.mark.parametrize("bad", _BAD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_a_dimension_the_render_family_refuses_is_refused_by_the_viewer(self, param, bad):
        render_verdict = _render_dims_verdict(param, bad)
        viewer_verdict = _verdict(
            lambda: _newton_open_viewer(_viewer_stub(has_display=True), viewer="gl", **{param: bad})
        )
        assert render_verdict == viewer_verdict == "refused", (
            f"{param}={bad!r}: render={render_verdict}, open_viewer={viewer_verdict}"
        )

    @pytest.mark.parametrize("good", _GOOD_DIMS)
    @pytest.mark.parametrize("param", ["width", "height"])
    def test_a_usable_dimension_is_accepted_by_both(self, param, good):
        """A floor, not a tightening, on both surfaces at once."""
        render_verdict = _render_dims_verdict(param, good)
        viewer_verdict = _verdict(
            lambda: _newton_open_viewer(_viewer_stub(has_display=True), viewer="gl", **{param: good})
        )
        assert render_verdict == viewer_verdict == "accepted", (
            f"{param}={good!r}: render={render_verdict}, open_viewer={viewer_verdict}"
        )

    def test_the_shared_rule_is_a_floor_and_not_a_ceiling(self):
        """A ray tracer has no framebuffer to overflow, so a huge size is legal.

        Pinned because ``_GOOD_DIMS`` already carries ``5000`` for the same
        reason: it makes the parity above a statement about the floor rather
        than about refusal in general.
        """
        huge = 10**9
        assert _render_dims_verdict("width", huge) == "accepted"
        assert (
            _verdict(lambda: _newton_open_viewer(_viewer_stub(has_display=True), viewer="gl", width=huge)) == "accepted"
        )


class TestNoNewtonDimensionSurfaceDrifts:
    """Every public Newton surface taking a pixel dimension reaches the domain.

    Structural rather than behavioural, because the defect this closes was a
    surface nobody had connected to a rule the module already applied twice. A
    method satisfies the guard by calling the shared helper itself or by handing
    the value to ``_resolve_camera_view``, the funnel the render family shares -
    so a fourth render entry point costs nothing, while a new one that sizes
    something of its own has to say why.
    """

    _FUNNEL = "_resolve_camera_view"
    _GUARD = "positive_count_error"
    #: The public surfaces taking a pixel dimension today. Pinned exactly so a
    #: mis-rooted scan reporting a clean sweep over nothing fails instead.
    _EXPECTED = frozenset({"add_camera", "render", "get_frame", "get_camera_params", "open_viewer"})

    @staticmethod
    def _classify(src: str) -> dict[str, str]:
        """Map each public dimension-taking method to how it reaches the domain."""
        found: dict[str, str] = {}
        for cls in ast.walk(ast.parse(src)):
            if not isinstance(cls, ast.ClassDef):
                continue
            for fn in ast.iter_child_nodes(cls):
                if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) or fn.name.startswith("_"):
                    continue
                args = fn.args
                names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
                if not {"width", "height"} & names:
                    continue
                calls = {
                    n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                    for n in ast.walk(fn)
                    if isinstance(n, ast.Call)
                }
                if TestNoNewtonDimensionSurfaceDrifts._GUARD in calls:
                    found[fn.name] = "guards"
                elif TestNoNewtonDimensionSurfaceDrifts._FUNNEL in calls:
                    found[fn.name] = "forwards"
                else:
                    found[fn.name] = "adrift"
        return found

    @property
    def _source(self) -> str:
        return Path(inspect.getfile(NewtonSimEngine)).read_text()

    def test_the_expected_surfaces_are_the_ones_found(self):
        assert set(self._classify(self._source)) == set(self._EXPECTED)

    def test_no_surface_is_adrift(self):
        adrift = sorted(name for name, how in self._classify(self._source).items() if how == "adrift")
        assert not adrift, f"{adrift} take a pixel dimension without reaching {self._GUARD}"

    def test_the_scan_detects_a_planted_surface(self):
        """Without this the sweep above could pass by matching nothing."""
        planted = self._source + (
            "\n\nclass _Planted:\n"
            "    def open_window(self, *, width: int = 1, height: int = 1) -> None:\n"
            "        self._build(width, height)\n"
        )
        assert self._classify(planted).get("open_window") == "adrift"


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
