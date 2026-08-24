"""An MJCF mesh asset's ``scale`` resolves through its ``<default>`` class.

MJCF lets a ``<default>`` class supply a mesh's ``scale``, exactly as it supplies
a ``<geom>``'s ``type`` and a ``<joint>``'s ``axis``::

    <default class="right_hand"><mesh scale="0.001 0.001 0.001"/></default>
    <asset><mesh class="right_hand" file="palm.obj"/></asset>

Read as the element's own attribute alone, such an asset reports the unit
fallback. That is a plausible value, so nothing downstream can tell "authored at
unit scale" from "the declared scale was not read": the asset is reported a
thousand times too large under a successful load. The reported scale is not
cosmetic - it rides onto the visual prim's xform as
:attr:`~strands_robots.simulation.isaac.loaders.SceneObject.mesh_scale`, and
:func:`~strands_robots.simulation.isaac.mesh_assets.mesh_aabb` measures the
object's collision proxy ``size`` through it, so a body whose only geometry is a
mesh reports a proxy scaled by the same factor.

Every expectation here is derived from MuJoCo rather than restated:
``mujoco.MjSpec`` exposes each mesh's own resolved ``scale``, which is the very
attribute this reader reports, so ``TestTheResolvedScaleIsTheOneMujocoResolves``
grades the loader against the compiler over each spelling. Note that a mesh
declaring no ``name`` is unnamed in the spec until ``compile()`` runs - the
compiler is what derives the name from the file - so the oracle compiles before
reading names.

``TestTheElementAndClassPrecedenceIsUnchanged`` pins the rules the fix must not
disturb: an element's own ``scale`` still wins over its class's, a model with no
``<default>`` at all is unaffected, and ``file``/``name`` stay element-only
reads, which is the format's rule rather than a simplification - MuJoCo's schema
refuses either attribute inside a ``<default><mesh>``.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import pytest

from strands_robots.simulation.isaac.loaders import (
    _parse_mjcf_mesh_assets,
    load_mjcf_scene_objects,
)

mujoco = pytest.importorskip("mujoco")

#: Tetrahedron, full extent (0.4, 0.3, 0.2) at unit scale, 4 vertices.
#: The three extents differ so a per-axis scale mix-up is visible in the size.
_TETRA_OBJ = "v 0 0 0\nv 0.4 0 0\nv 0 0.3 0\nv 0 0 0.2\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"

#: That extent, so no expectation below restates it.
_TETRA_EXTENT = (0.4, 0.3, 0.2)

#: The scale a reader that ignores the class reports. Every expectation below
#: differs from it, which is what makes a passing test evidence of a real read.
_UNIT = (1.0, 1.0, 1.0)

#: Deliberately anisotropic: a reader that resolved the class but transposed or
#: broadcast an axis would still disagree with the compiler.
_CLASS_SCALE = "0.2 0.3 0.5"
_CLASS_SCALE_T = (0.2, 0.3, 0.5)


def _write_model(root: pathlib.Path, body: str, **fragments: str) -> str:
    """Write ``body`` as ``model.xml`` (plus any fragments) and return its path."""
    (root / "widget.obj").write_text(_TETRA_OBJ, encoding="utf-8")
    for name, text in fragments.items():
        (root / f"{name}.xml").write_text(text, encoding="utf-8")
    top = root / "model.xml"
    top.write_text(body, encoding="utf-8")
    return str(top)


def _reported_scale(path: str, mesh_name: str = "widget_mesh") -> tuple[float, float, float]:
    """The loader's reported scale for one mesh asset."""
    root = ET.parse(path).getroot()
    registry = _parse_mjcf_mesh_assets(root, str(pathlib.Path(path).parent))
    assert mesh_name in registry, f"{mesh_name!r} not in {sorted(registry)}"
    return registry[mesh_name][1]


def _mujoco_scale(path: str, mesh_name: str = "widget_mesh") -> tuple[float, float, float]:
    """MuJoCo's own resolved ``scale`` for one mesh asset.

    ``compile()`` runs first because a mesh declaring no ``name`` carries an
    empty one in the parsed spec - the compiler derives it from the file - and
    an oracle keyed on empty names would collapse every such mesh into one entry
    and silently compare nothing.
    """
    spec = mujoco.MjSpec.from_file(path)
    spec.compile()
    by_name = {m.name: tuple(round(float(v), 12) for v in m.scale) for m in spec.meshes}
    assert "" not in by_name, "a mesh is still unnamed after compile()"
    assert mesh_name in by_name, f"{mesh_name!r} not in {sorted(by_name)}"
    return by_name[mesh_name]  # type: ignore[return-value]


#: One movable body whose only geometry is the mesh, so the reported proxy
#: ``size`` is measured through the reported scale.
def _body(mesh_ref: str = "widget_mesh") -> str:
    return (
        "  <worldbody>\n"
        '    <body name="widget" pos="0 0 0"><freejoint/>\n'
        f'      <geom type="mesh" mesh="{mesh_ref}" group="1"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
    )


_BODY = _body()


def _model(defaults: str, asset: str, mesh_ref: str = "widget_mesh") -> str:
    return f"<mujoco>\n{defaults}  <asset>{asset}</asset>\n{_body(mesh_ref)}</mujoco>\n"


#: ``(label, <default> block, <mesh> element, expected scale)`` - each a model
#: MuJoCo compiles, pinned in ``TestTheResolvedScaleIsTheOneMujocoResolves``.
_CLASS_SPELLINGS = [
    (
        "a named class the element names",
        f'  <default><default class="mm"><mesh scale="{_CLASS_SCALE}"/></default></default>\n',
        '<mesh name="widget_mesh" class="mm" file="widget.obj"/>',
        _CLASS_SCALE_T,
    ),
    (
        "the root class, which the element reaches by naming nothing",
        f'  <default><mesh scale="{_CLASS_SCALE}"/></default>\n',
        '<mesh name="widget_mesh" file="widget.obj"/>',
        _CLASS_SCALE_T,
    ),
    (
        "the root class spelled main",
        f'  <default><mesh scale="{_CLASS_SCALE}"/></default>\n',
        '<mesh name="widget_mesh" class="main" file="widget.obj"/>',
        _CLASS_SCALE_T,
    ),
    (
        "the innermost class of a nested chain",
        '  <default><default class="outer"><mesh scale="9 9 9"/>\n'
        f'    <default class="inner"><mesh scale="{_CLASS_SCALE}"/></default></default></default>\n',
        '<mesh name="widget_mesh" class="inner" file="widget.obj"/>',
        _CLASS_SCALE_T,
    ),
    (
        "an enclosing class, inherited by a nested one that declares no mesh",
        f'  <default><default class="outer"><mesh scale="{_CLASS_SCALE}"/>\n'
        '    <default class="inner"/></default></default>\n',
        '<mesh name="widget_mesh" class="inner" file="widget.obj"/>',
        _CLASS_SCALE_T,
    ),
]


class TestAMeshAssetScaleComesFromItsDefaultClass:
    """Each spelling by which a class supplies a mesh's scale is read."""

    @pytest.mark.parametrize(
        ("label", "defaults", "asset", "expected"),
        [pytest.param(*c, id=c[0].replace(" ", "_")) for c in _CLASS_SPELLINGS],
    )
    def test_the_class_declared_scale_is_reported(
        self, tmp_path: pathlib.Path, label: str, defaults: str, asset: str, expected: tuple[float, float, float]
    ) -> None:
        assert expected != _UNIT, "premise: the expectation differs from the unit fallback"
        path = _write_model(tmp_path, _model(defaults, asset))
        assert _reported_scale(path) == pytest.approx(expected), label

    def test_a_class_declared_in_an_included_fragment_is_read(self, tmp_path: pathlib.Path) -> None:
        """``<default>`` is model-global, so the class may live in a fragment."""
        path = _write_model(
            tmp_path,
            "<mujoco>\n"
            '  <include file="classes.xml"/>\n'
            '  <asset><mesh name="widget_mesh" class="mm" file="widget.obj"/></asset>\n'
            f"{_BODY}</mujoco>\n",
            classes=f'<mujoco><default><default class="mm"><mesh scale="{_CLASS_SCALE}"/></default></default></mujoco>',
        )
        assert _reported_scale(path) == pytest.approx(_CLASS_SCALE_T)

    def test_an_unnamed_mesh_is_keyed_by_its_basename_and_still_scaled(self, tmp_path: pathlib.Path) -> None:
        """A ``<mesh>`` may declare a class and no name, which is the common shape.

        Most of the affected shipped assets are written this way - the class
        carries the unit conversion and each mesh names only its file - so the
        two rules have to hold together: the entry is keyed by the file's
        basename AND scaled by its class.
        """
        defaults = f'  <default><default class="mm"><mesh scale="{_CLASS_SCALE}"/></default></default>\n'
        path = _write_model(tmp_path, _model(defaults, '<mesh class="mm" file="widget.obj"/>', mesh_ref="widget"))

        assert _reported_scale(path, "widget") == pytest.approx(_CLASS_SCALE_T)
        assert _reported_scale(path, "widget") == pytest.approx(_mujoco_scale(path, "widget"))

    def test_the_scale_reaches_the_reported_object_and_its_proxy_size(self, tmp_path: pathlib.Path) -> None:
        """Through the public loader: the object's scale AND its proxy extent.

        The body carries no analytic collision geometry, so the reported
        ``size`` is the mesh's own extent measured through the reported scale -
        the tetrahedron's (0.4, 0.3, 0.2) times the class's scale.
        """
        defaults = f'  <default><default class="mm"><mesh scale="{_CLASS_SCALE}"/></default></default>\n'
        asset = '<mesh name="widget_mesh" class="mm" file="widget.obj"/>'
        path = _write_model(tmp_path, _model(defaults, asset))

        (obj,) = load_mjcf_scene_objects(path)

        assert obj.mesh_scale == pytest.approx(_CLASS_SCALE_T)
        expected_size = tuple(e * s for e, s in zip(_TETRA_EXTENT, _CLASS_SCALE_T, strict=True))
        assert obj.size == pytest.approx(expected_size)
        assert obj.size != pytest.approx(_TETRA_EXTENT), "premise: the proxy differs from the unit-scale one"


class TestTheResolvedScaleIsTheOneMujocoResolves:
    """The loader's reported scale is graded against the compiler's own."""

    @pytest.mark.parametrize(
        ("label", "defaults", "asset", "expected"),
        [pytest.param(*c, id=c[0].replace(" ", "_")) for c in _CLASS_SPELLINGS],
    )
    def test_every_class_spelling_matches_mujoco(
        self, tmp_path: pathlib.Path, label: str, defaults: str, asset: str, expected: tuple[float, float, float]
    ) -> None:
        path = _write_model(tmp_path, _model(defaults, asset))
        truth = _mujoco_scale(path)
        assert truth == pytest.approx(expected), f"premise: the fixture means what it says ({label})"
        assert _reported_scale(path) == pytest.approx(truth), label

    def test_an_element_scale_matches_mujoco(self, tmp_path: pathlib.Path) -> None:
        path = _write_model(
            tmp_path, _model("", f'<mesh name="widget_mesh" file="widget.obj" scale="{_CLASS_SCALE}"/>')
        )
        assert _reported_scale(path) == pytest.approx(_mujoco_scale(path))


class TestTheElementAndClassPrecedenceIsUnchanged:
    """The rules the class resolution must not disturb."""

    def test_the_elements_own_scale_wins_over_its_class(self, tmp_path: pathlib.Path) -> None:
        defaults = f'  <default><default class="mm"><mesh scale="{_CLASS_SCALE}"/></default></default>\n'
        asset = '<mesh name="widget_mesh" class="mm" file="widget.obj" scale="4 4 4"/>'
        path = _write_model(tmp_path, _model(defaults, asset))
        assert _reported_scale(path) == pytest.approx((4.0, 4.0, 4.0))
        assert _reported_scale(path) == pytest.approx(_mujoco_scale(path))

    def test_a_model_with_no_defaults_reports_the_element_scale(self, tmp_path: pathlib.Path) -> None:
        path = _write_model(tmp_path, _model("", '<mesh name="widget_mesh" file="widget.obj" scale="0.5 0.5 0.5"/>'))
        assert _reported_scale(path) == pytest.approx((0.5, 0.5, 0.5))

    def test_a_mesh_declaring_no_scale_anywhere_reports_unit(self, tmp_path: pathlib.Path) -> None:
        path = _write_model(tmp_path, _model("", '<mesh name="widget_mesh" file="widget.obj"/>'))
        assert _reported_scale(path) == pytest.approx(_UNIT)
        assert _mujoco_scale(path) == pytest.approx(_UNIT)

    @pytest.mark.parametrize(("kind", "attribute"), [("geom", "scale"), ("mesh", "size")])
    def test_mujoco_keeps_the_two_kinds_attribute_sets_disjoint(
        self, tmp_path: pathlib.Path, kind: str, attribute: str
    ) -> None:
        """Why the mesh class map is collected separately from the geom one.

        One ``<default>`` class carries a separate attribute set per element
        kind, and MuJoCo enforces it: neither ``scale`` on a ``<geom>`` nor
        ``size`` on a ``<mesh>`` is a legal declaration, so resolving a mesh
        against the geom map could only ever lose the scale - never borrow a
        geom's. That is the shape the regression classes above measure.
        """
        defaults = f'  <default><{kind} {attribute}="0.7 0.7 0.7"/></default>\n'
        path = _write_model(tmp_path, _model(defaults, '<mesh name="widget_mesh" file="widget.obj"/>'))
        with pytest.raises(ValueError, match=f"unrecognized attribute: '{attribute}'"):
            mujoco.MjSpec.from_file(path)

    def test_a_geom_class_sharing_the_name_does_not_change_the_scale(self, tmp_path: pathlib.Path) -> None:
        """One class name, two element kinds, one scale."""
        defaults = (
            f'  <default><default class="mm"><geom type="mesh" group="1"/>'
            f'<mesh scale="{_CLASS_SCALE}"/></default></default>\n'
        )
        asset = '<mesh name="widget_mesh" class="mm" file="widget.obj"/>'
        path = _write_model(tmp_path, _model(defaults, asset))
        assert _reported_scale(path) == pytest.approx(_CLASS_SCALE_T)
        assert _reported_scale(path) == pytest.approx(_mujoco_scale(path))

    def test_a_class_no_default_declares_contributes_nothing(self, tmp_path: pathlib.Path) -> None:
        """Naming the offending class is MuJoCo's report to make, not this reader's."""
        asset = '<mesh name="widget_mesh" class="absent" file="widget.obj"/>'
        path = _write_model(tmp_path, _model("", asset))
        assert _reported_scale(path) == pytest.approx(_UNIT)

    @pytest.mark.parametrize("attribute", ["file", "name"])
    def test_mujoco_refuses_an_asset_attribute_inside_a_default(self, tmp_path: pathlib.Path, attribute: str) -> None:
        """Why ``file`` and ``name`` stay element-only reads: the format says so."""
        defaults = f'  <default><mesh {attribute}="widget.obj"/></default>\n'
        path = _write_model(tmp_path, _model(defaults, '<mesh name="widget_mesh" file="widget.obj"/>'))
        with pytest.raises(ValueError, match=f"unrecognized attribute: '{attribute}'"):
            mujoco.MjSpec.from_file(path)
