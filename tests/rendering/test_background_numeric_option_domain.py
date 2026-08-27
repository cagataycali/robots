# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A background's alignment *numbers* are checked, not coerced with a bare ``float()``.

``strands_robots.rendering`` closed its posture flags on
:func:`~strands_robots.utils.boolean_flag_error`, and that domain is documented
as the one "for a flag that selects a posture rather than scaling a quantity".
The quantities were the other half. Every scalar number
:class:`~strands_robots.rendering.PanoramaBackground` and
:class:`~strands_robots.rendering.GsplatBackground` take was handed to a bare
``float()``, which accepts ``nan`` and ``inf``, so a non-finite angle, radius,
height or fraction was stored and read as if it were a measurement.

Two consequences follow, and both are pinned here.

``PanoramaBackground.rotation_deg`` builds the ``Rz`` every world ray is turned
by. A non-finite yaw turned each direction into ``nan``, the equirectangular
lookup sampled nothing, and ``render`` returned a **uniformly black** backdrop
-- ``rgb.mean() == 0.0`` with a single distinct value -- while reporting no
error at all. That is the failure mode :class:`GsplatBackground`'s own path
comment warns about from the other side: in app contexts the photoreal
background sits inside a catch-all that demotes it to a procedural fallback, and
a silent black frame never raises, so the fallback cannot fire.

``GsplatBackground``'s numbers feed the fitted ``world_from_gs``. A non-finite
``up_sign``, ``yaw_deg``, ``radius``, ``floor_z`` or ``backdrop_radius``
produced a 4x4 with non-finite cells, so every gaussian was placed nowhere;
``min_opacity`` was worse than nowhere because ``nan > 0`` is ``False``, which
skips the opacity filter the value asks to apply.

The domain is :func:`~strands_robots.utils.finite_number_error`, the shared one
for a signed physical quantity a caller supplies verbatim, and it is the
authority these tests parametrize over, so a spelling added there is covered
here without an edit. It is the same rule
:func:`~strands_robots.rendering.compositor._shadow_plane_z_error` states for a
plane height -- "only a finite number can be honored: a non-finite plane
intersects no ray and the shadow pass would silently never fire".

Deliberately not in scope: **bounds**. Whether ``min_opacity`` belongs in
``[0, 1]`` and ``floor_pct`` in ``[0, 100]`` is a policy question this change
does not answer, and ``numpy.percentile`` already refuses a non-finite
``floor_pct`` loudly on its own. Vector options (``center``, ``backdrop_center``,
``up_axis``, ``major_axis``) are tuples and want an element-wise domain rather
than this scalar one.

Dependency-free: both constructors defer every ``gsplat``/``torch`` import to
the first render, the panorama is pure NumPy, and the transform consequence is
measured through the pure-NumPy skybox fit -- nothing here needs the ``sim-gs``
extra.
"""

import ast
import inspect
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import GsplatBackground, PanoramaBackground
from strands_robots.rendering import backgrounds as backgrounds_module
from strands_robots.rendering.backgrounds import _fit_skybox_transform
from strands_robots.rendering.compositor import CameraParams
from strands_robots.utils import finite_number_error

#: The shared domain's name, as the module under test must spell it.
SHARED_DOMAIN = "finite_number_error"

#: Annotations that mark a constructor parameter as a caller-supplied scalar
#: number. A ``| None`` spelling carries a sentinel the domain must let past.
SCALAR_ANNOTATIONS: frozenset[str] = frozenset({"float", "float | None"})

#: Values that are not a finite number. Each was accepted before this change.
NON_FINITE_SPELLINGS: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "90",
    None,
    [90.0],
    10**400,
]

#: Numbers a caller legitimately supplies, including both signs and a NumPy
#: scalar read out of a config or a policy action.
USABLE_NUMBERS: list[Any] = [0.0, 1.0, -1.0, 90.0, -90.0, 720.0, 2, np.float32(45.0)]


def _scalar_numeric_params(cls: type) -> list[tuple[str, str]]:
    """The ``(name, annotation)`` of every scalar-number ctor parameter of ``cls``.

    Derived from the annotation rather than a hardcoded list, so a background
    that grows a tenth number is held to the same rule the hour it lands.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    found: list[tuple[str, str]] = []
    for arg in list(fn.args.args)[1:] + list(fn.args.kwonlyargs):
        if arg.annotation is None:
            continue
        annotation = ast.unparse(arg.annotation)
        if annotation in SCALAR_ANNOTATIONS:
            found.append((arg.arg, annotation))
    return found


def _guarded_params(cls: type) -> set[str]:
    """Every parameter name handed to the shared domain inside ``cls.__init__``.

    Reads the guard's own argument rather than the stored attribute: two classes
    holding equal numbers is exactly what the unguarded state looked like, so
    the call is the property. Both spellings the module uses are understood -- a
    name passed directly, and a name supplied by a ``for`` over a tuple of
    ``(name, value)`` pairs, which is how the posture flags beside these are
    checked. The context argument is not a parameter name, so it is read from
    its own position and never swept in as one.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    guarded: set[str] = set()
    for statement in fn.body:
        calls = [
            node
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == SHARED_DOMAIN
        ]
        if not calls:
            continue
        if isinstance(statement, ast.For) and isinstance(statement.iter, ast.Tuple):
            for pair in statement.iter.elts:
                if isinstance(pair, ast.Tuple) and pair.elts:
                    name = pair.elts[0]
                    if isinstance(name, ast.Constant) and isinstance(name.value, str):
                        guarded.add(name.value)
            continue
        for call in calls:
            if len(call.args) < 2:
                continue
            name = call.args[1]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                guarded.add(name.value)
    return guarded


def _sentinel_params(cls: type) -> set[str]:
    """Scalar parameters whose annotation admits ``None``."""
    return {name for name, ann in _scalar_numeric_params(cls) if ann == "float | None"}


#: The backgrounds that take caller-supplied numbers, discovered by annotation.
NUMERIC_BACKGROUNDS: list[type] = [PanoramaBackground, GsplatBackground]


@pytest.fixture
def scene_ply(tmp_path):
    """An existing placeholder scene file: construction validates the path but
    defers the decode to the first render, so no ``sim-gs`` extra is needed."""
    path = tmp_path / "scene.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    return path


def _build(cls: type, scene_ply, **overrides: Any) -> Any:
    """Construct ``cls`` with its required arguments plus ``overrides``."""
    if cls is GsplatBackground:
        return cls(scene_ply, **overrides)
    return cls(**overrides)


def _camera() -> CameraParams:
    """A small pinhole camera, big enough for the equirect lookup to vary."""
    intrinsics = np.array([[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.5, 0.3]
    return CameraParams(width=160, height=120, K=intrinsics, T_world_cam=pose, znear=0.01, zfar=50.0)


class TestEveryScalarNumberIsCheckedOnTheSharedDomain:
    """The derived inventory: no background keeps a number outside the domain."""

    @pytest.mark.parametrize("cls", NUMERIC_BACKGROUNDS, ids=lambda c: c.__name__)
    def test_every_scalar_numeric_parameter_reaches_the_shared_domain(self, cls: type) -> None:
        declared = {name for name, _ in _scalar_numeric_params(cls)}
        assert declared, f"{cls.__name__} declares no scalar number, so this rule grades nothing"
        assert declared - _guarded_params(cls) == set(), (
            f"{cls.__name__} takes a caller-supplied number that never reaches {SHARED_DOMAIN}"
        )

    @pytest.mark.parametrize("cls", NUMERIC_BACKGROUNDS, ids=lambda c: c.__name__)
    def test_the_domain_is_not_invoked_for_a_parameter_that_is_not_one(self, cls: type) -> None:
        declared = {name for name, _ in _scalar_numeric_params(cls)}
        assert _guarded_params(cls) - declared == set(), (
            f"{cls.__name__} guards a name that is not one of its scalar numbers"
        )

    def test_the_module_reads_the_shared_domain_rather_than_restating_it(self) -> None:
        # A local re-implementation holding today's identical rule is exactly
        # what the unguarded state looked like, so the import is the property.
        imported: set[str] = set()
        for node in ast.walk(ast.parse(inspect.getsource(backgrounds_module))):
            if isinstance(node, ast.ImportFrom) and node.module == "strands_robots.utils":
                imported |= {alias.name for alias in node.names}
        assert SHARED_DOMAIN in imported


class TestANonFiniteYawRenderedAnEntirelyBlackBackdrop:
    """The panorama consequence, measured through the shipped render."""

    @pytest.mark.parametrize("value", NON_FINITE_SPELLINGS, ids=repr)
    def test_construction_refuses_the_value_naming_the_parameter(self, value: Any) -> None:
        with pytest.raises(ValueError, match="PanoramaBackground: rotation_deg must be"):
            PanoramaBackground(rotation_deg=value)

    def test_the_refusal_precedes_the_coercion_that_swallowed_the_value(self) -> None:
        # ``float(np.deg2rad(nan))`` is a perfectly good float, so a guard placed
        # after the coercion would inspect a value the defect has already made
        # indistinguishable from a measurement.
        source = inspect.getsource(PanoramaBackground.__init__)
        assert SHARED_DOMAIN in source, f"the constructor never reaches {SHARED_DOMAIN}"
        assert source.index(SHARED_DOMAIN) < source.index("np.deg2rad")


class TestANonFiniteAlignmentNumberPoisonedTheFittedTransform:
    """The gsplat consequence, measured through the pure-NumPy skybox fit."""

    @pytest.mark.parametrize("param", ["backdrop_radius", "yaw_deg", "radius", "floor_z", "min_opacity", "floor_pct"])
    @pytest.mark.parametrize("value", NON_FINITE_SPELLINGS, ids=repr)
    def test_construction_refuses_the_value_naming_the_parameter(self, scene_ply, param: str, value: Any) -> None:
        with pytest.raises(ValueError, match=f"GsplatBackground: {param} must be"):
            _build(GsplatBackground, scene_ply, **{param: value})

    @pytest.mark.parametrize("param", ["up_sign", "clip_below"])
    @pytest.mark.parametrize("value", [v for v in NON_FINITE_SPELLINGS if v is not None], ids=repr)
    def test_a_sentinel_bearing_number_is_refused_like_the_rest(self, scene_ply, param: str, value: Any) -> None:
        with pytest.raises(ValueError, match=f"GsplatBackground: {param} must be"):
            _build(GsplatBackground, scene_ply, **{param: value})


class TestWhatTheRefusalMustNotCost:
    """Over-reach controls: every number the fit can honor is still accepted."""

    def test_a_usable_yaw_still_rotates_rather_than_blanking_the_backdrop(self) -> None:
        camera = _camera()
        unrotated, _ = PanoramaBackground(rotation_deg=0.0).render(camera)
        rotated, _ = PanoramaBackground(rotation_deg=90.0).render(camera)
        # The consequence the refusal exists to prevent: a rotated backdrop is a
        # different image, not an absent one.
        assert not np.array_equal(unrotated, rotated)
        for frame in (unrotated, rotated):
            assert frame.mean() > 0.0
            assert len(np.unique(frame)) > 1

    @pytest.mark.parametrize("value", USABLE_NUMBERS, ids=repr)
    def test_a_usable_yaw_is_still_accepted(self, value: Any) -> None:
        assert PanoramaBackground(rotation_deg=value) is not None

    @pytest.mark.parametrize("param", ["backdrop_radius", "yaw_deg", "radius", "floor_z", "min_opacity", "floor_pct"])
    def test_a_usable_alignment_number_is_still_accepted(self, scene_ply, param: str) -> None:
        for value in (0.0, 1.0, -1.0, np.float32(2.5)):
            assert _build(GsplatBackground, scene_ply, **{param: value}) is not None

    @pytest.mark.parametrize("param", ["up_sign", "clip_below"])
    def test_the_sentinel_still_selects_its_documented_default(self, scene_ply, param: str) -> None:
        # ``None`` is the documented spelling of "auto-detect" / "drop nothing",
        # so the domain must let it past rather than reading it as a bad number.
        background = _build(GsplatBackground, scene_ply, **{param: None})
        assert getattr(background, f"_{param}") is None

    def test_the_defaults_the_signature_declares_are_all_usable(self, scene_ply) -> None:
        # A guard that refused its own default would be caught nowhere else.
        assert PanoramaBackground() is not None
        assert _build(GsplatBackground, scene_ply) is not None


class TestThePremisesTheseTestsRestOn:
    """The shared domain and the sentinel set behave as the rule assumes."""

    @pytest.mark.parametrize("value", NON_FINITE_SPELLINGS, ids=repr)
    def test_the_shared_domain_refuses_every_spelling_graded_above(self, value: Any) -> None:
        assert finite_number_error(value, "p", "C") is not None

    @pytest.mark.parametrize("value", USABLE_NUMBERS, ids=repr)
    def test_the_shared_domain_accepts_every_number_graded_above(self, value: Any) -> None:
        assert finite_number_error(value, "p", "C") is None

    def test_both_classes_carry_a_sentinel_free_and_a_sentinel_bearing_number(self) -> None:
        # Without both kinds present the two-loop shape in the source would be
        # untested on one of its branches.
        assert _sentinel_params(GsplatBackground) == {"up_sign", "clip_below"}
        assert _sentinel_params(PanoramaBackground) == set()

    @pytest.mark.parametrize("param", ["up_sign", "yaw_deg", "radius", "floor_z"])
    def test_the_fit_returns_a_non_finite_transform_for_a_non_finite_number(self, param: str) -> None:
        # The premise for the constructor's refusal: this is what the fit does
        # with the value, so refusing it up front is the only way to keep the
        # transform a placement rather than a hole.
        rng = np.random.default_rng(7)
        means = rng.normal(size=(400, 3)) * np.array([2.0, 2.0, 0.7])
        assert np.isfinite(_fit_skybox_transform(means)).all()
        poisoned = _fit_skybox_transform(means, **{param: float("nan")})
        assert not np.isfinite(poisoned).all()
