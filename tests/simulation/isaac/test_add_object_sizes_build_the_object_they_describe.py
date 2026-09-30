"""An Isaac ``add_object`` size builds the object it describes, or is refused by name.

Measured on one L40S, Isaac Sim 6.1, a ``cylinder`` with the 3-component size the
published tool schema documents - ``[diameter, unused, full height]``, the MuJoCo
backend's layout - was read here as ``[radius, height]``:

* ``[0.04, 0, 0.06]`` built a zero-height collider that fell to z = -19.9 m in
  2 s, under ``status="success"`` (MuJoCo: rests at z = 0.0299);
* ``[0.04, 0.04, 0.06]`` built a 4 cm tall cylinder of radius 4 cm (MuJoCo: 6 cm
  tall, radius 2 cm);
* a box with a zero extent failed deep in USD with "Non-positive determinant
  (left-handed or null coordinate frame) in rotation matrix".

The two-component ``[radius, height]`` form this backend documented keeps its
meaning. Unit-level: the prim constructor is stood in, so what is graded is the
radius and height it is handed, and which sizes never reach it.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacSimulation,
    _primitive_size_error,
    _round_shape_dims,
)
from tests.simulation.test_object_size_domain_across_backends import _isaac_recording  # noqa: E402


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


class TestARoundShapeReadsBothLayouts:
    @pytest.mark.parametrize(
        ("size", "dims"),
        [
            ([0.04, 0.0, 0.06], (0.02, 0.06)),  # MuJoCo / tool schema: diameter, unused, height
            ([0.04, 0.04, 0.06], (0.02, 0.06)),
            ([0.03, 0.08], (0.03, 0.08)),  # this backend's own radius, height
            ([0.03], (0.03, 0.10)),  # trailing default kept
            (None, (0.05, 0.10)),
        ],
    )
    def test_radius_and_height(self, size: list[float] | None, dims: tuple[float, float]) -> None:
        assert _round_shape_dims(size) == pytest.approx(dims)


class TestAZeroConsumedExtentIsRefused:
    @pytest.mark.parametrize(
        ("shape", "size", "named"),
        [
            ("cylinder", [0.04, 0.0], "height=0"),
            ("cylinder", [0.0, 0.0, 0.06], "radius=0"),
            ("capsule", [0.04, 0.5, 0.0], "height=0"),
            ("box", [0.1, 0.0, 0.1], "y=0"),
            ("box", [0.1, 0.1, -0.2], "z=-0.2"),
            ("sphere", [0.0], "radius=0"),
        ],
    )
    def test_it_names_the_component_and_builds_nothing(self, shape: str, size: list[float], named: str) -> None:
        stub, seen = _isaac_recording()
        result = IsaacSimulation.add_object(stub, "thing", shape=shape, size=size)
        assert result["status"] == "error"
        assert named in _text(result) and "Nothing was added" in _text(result)
        assert seen["construct"] == 0 and seen["scene_add"] == 0

    @pytest.mark.parametrize(
        ("shape", "size"),
        [("cylinder", [0.04, 0.0, 0.06]), ("cylinder", [0.03, 0.08]), ("box", [0.1]), ("sphere", [0.02])],
    )
    def test_a_buildable_size_passes(self, shape: str, size: list[float]) -> None:
        assert _primitive_size_error(shape, size) is None


class TestThePrimIsBuiltWithTheResolvedDims:
    def test_the_cylinder_constructor_gets_radius_and_height(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        seen: dict[str, Any] = {}

        class _Prim:
            def __init__(self, **kwargs: Any) -> None:
                seen.update(kwargs)

        objects = types.ModuleType("isaacsim.core.api.objects")
        for cls in (
            "DynamicCapsule",
            "DynamicCuboid",
            "DynamicCylinder",
            "DynamicSphere",
            "FixedCapsule",
            "FixedCuboid",
            "FixedCylinder",
            "FixedSphere",
        ):
            setattr(objects, cls, _Prim)
        for name in ("isaacsim", "isaacsim.core", "isaacsim.core.api"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "isaacsim.core.api.objects", objects)

        engine = IsaacSimulation.__new__(IsaacSimulation)
        _, resolved = engine._construct_shape_prim(
            shape="cylinder",
            prim_path="/World/Objects/can",
            name="can",
            position=[0.0, 0.0, 0.1],
            orientation=[1.0, 0.0, 0.0, 0.0],
            size=[0.04, 0.0, 0.06],
            color=None,
            mass=0.1,
            is_static=False,
        )
        assert seen["radius"] == pytest.approx(0.02) and seen["height"] == pytest.approx(0.06)
        assert resolved == pytest.approx([0.02, 0.06])
