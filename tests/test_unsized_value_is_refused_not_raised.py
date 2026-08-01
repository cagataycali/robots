"""A caller value whose length cannot be read is reported, never raised.

Every validator that accepts a vector first asks "how many components is this?".
The obvious spelling - ``hasattr(value, "__len__")`` followed by ``len(value)`` -
is unsafe for a value class this library receives routinely: a 0-d numpy array
(``np.array(0.5)``, the result of a reduction such as ``np.mean(...)``) and a 0-d
torch tensor both *declare* ``__len__`` and then raise from it. The ``hasattr``
probe passes and the ``len()`` call escapes with a bare ``len() of unsized
object`` naming neither the parameter nor the method.

That escape matters because the surfaces doing the probing all publish a
no-raise contract: the MuJoCo agent-tool router returns a structured error for
every rejected parameter, :meth:`SimEngine.get_world_point` documents its
structural checks as being there "to keep the never-raises envelope", and
:func:`strands_robots.rendering.video.mjpeg_frames` documents ``ValueError`` with
an actionable message as its only failure mode for a malformed ``size``.

:func:`strands_robots.utils.sequence_length` is the single owner of the rule -
it answers "no readable length" for a 0-d array and for a plain scalar alike -
and these tests pin every surface that reads a caller-supplied length through
it, plus the accepted values that must keep working.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.rendering.video import mjpeg_frames
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.utils import sequence_length

# A 0-d array: declares ``__len__``, raises from it, holds exactly one scalar.
UNSIZED = np.array(0.5)


# A minimal actuated arm: enough for send_action to reach the action coercion,
# with no downloaded asset and no rendering (the model is compiled, never drawn).
_ARM_MJCF = """<mujoco model="unsized_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="link" pos="0 0 0.05">
        <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="pan_act" joint="pan" kp="50" ctrlrange="-2 2"/></actuator>
</mujoco>
"""


def _call(target: Any, **kwargs: Any) -> Any:
    """Invoke ``target`` with values whose *shape* is the subject of the test.

    These parameters are annotated for the shapes a caller *should* pass
    (``pixels`` as ``Sequence[Sequence[SupportsFloat]]``, ``size`` as a
    ``tuple[int, int]``), while the point of this module is what happens for the
    shapes a caller *does* pass - a 0-d NumPy array, and a correctly sized NumPy
    array that those annotations do not describe either. Routing every such call
    through one ``**kwargs: Any`` funnel states that intent once instead of
    scattering per-call type suppressions.
    """
    return target(**kwargs)


def _text(result: dict[str, Any]) -> str:
    """Concatenate the text blocks of an agent-tool envelope."""
    return " ".join(block["text"] for block in result.get("content", []) if "text" in block)


# --------------------------------------------------------------------------- #
# The engine premise the whole rule rests on.
# --------------------------------------------------------------------------- #
class TestUnsizedValuePremise:
    """Pin the numpy/torch behaviour that makes the ``hasattr`` probe unsafe."""

    def test_a_zero_dimensional_array_declares_a_length_then_refuses_it(self) -> None:
        """``hasattr`` says yes and ``len()`` raises - the whole defect in two lines."""
        assert hasattr(UNSIZED, "__len__")
        with pytest.raises(TypeError):
            len(UNSIZED)

    def test_a_zero_dimensional_tensor_behaves_the_same_way(self) -> None:
        """torch shares the property, so the rule is not numpy-specific."""
        torch = pytest.importorskip("torch")
        scalar = torch.tensor(0.5)
        assert hasattr(scalar, "__len__")
        with pytest.raises(TypeError):
            len(scalar)

    def test_a_numpy_scalar_declares_no_length_at_all(self) -> None:
        """The other half of the domain: ``len()`` raises ``TypeError`` here too.

        Both spellings answer a validator's question identically - this value
        carries no component count - which is why one branch covers them.
        """
        assert not hasattr(np.float64(0.5), "__len__")
        with pytest.raises(TypeError):
            len(np.float64(0.5))  # type: ignore[arg-type]


class TestSequenceLength:
    """The shared owner of the rule."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([1.0, 2.0, 3.0], 3),
            ((1.0, 2.0), 2),
            (np.array([1.0, 2.0, 3.0]), 3),
            (np.zeros((2, 3)), 2),
            ("abc", 3),
            ({"a": 1}, 1),
            ([], 0),
        ],
    )
    def test_reports_the_component_count_of_a_sized_value(self, value: Any, expected: int) -> None:
        """A readable length is returned unchanged."""
        assert sequence_length(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            np.array(0.5),
            np.array(True),
            np.float64(0.5),
            np.int64(3),
            0.5,
            None,
            object(),
        ],
        ids=["zero_d_float", "zero_d_bool", "np_float64", "np_int64", "float", "none", "object"],
    )
    def test_reports_none_for_a_value_without_a_readable_length(self, value: Any) -> None:
        """No readable length is ``None``, never an exception."""
        assert sequence_length(value) is None


# --------------------------------------------------------------------------- #
# The agent-tool router: one structured error per rejected vector parameter.
# --------------------------------------------------------------------------- #
# Every router vector param, paired with a method whose signature declares it
# (the router rejects a parameter the target method does not accept before it
# ever reaches the dimension check).
_VECTOR_PARAM_OWNERS: dict[str, tuple[str, dict[str, Any]]] = {
    "position": ("add_object", {"name": "crate"}),
    "target": ("add_camera", {"name": "cam"}),
    "origin": ("raycast", {"direction": [0.0, 0.0, -1.0]}),
    "force": ("apply_force", {"body_name": "crate"}),
    "torque": ("apply_force", {"body_name": "crate"}),
    "gravity": ("set_gravity", {}),
    "direction": ("raycast", {"origin": [0.0, 0.0, 1.0]}),
    "point": ("apply_force", {"body_name": "crate"}),
    "orientation": ("add_object", {"name": "crate"}),
    "color": ("add_object", {"name": "crate"}),
}


class TestAgentToolRouterVectorParams:
    """No dispatched vector parameter may escape the envelope on a length probe."""

    def test_every_router_vector_param_is_covered_here(self) -> None:
        """Exhaustiveness: a twelfth vector param must be pinned too.

        ``_FIELD_ALIASES`` entries are remapped to their canonical name before
        the dimension check runs, so they are covered by the name they alias.
        """
        aliases = set(MuJoCoSimEngine._FIELD_ALIASES)
        assert set(_VECTOR_PARAM_OWNERS) == set(MuJoCoSimEngine._VECTOR_PARAM_LENGTHS) - aliases

    @pytest.mark.parametrize("param", sorted(_VECTOR_PARAM_OWNERS))
    def test_an_unsized_vector_is_reported_through_the_envelope(self, param: str) -> None:
        """The caller gets a structured error naming the parameter, not a raise."""
        action, extra = _VECTOR_PARAM_OWNERS[param]
        engine = MuJoCoSimEngine(tool_name="unsized_router", mesh=False)
        signature = inspect.signature(getattr(MuJoCoSimEngine, action))
        _, error = engine._validate_and_build_kwargs(action, action, signature, {**extra, param: UNSIZED})
        assert error is not None, f"{action}({param}=<0-d array>) was accepted"
        assert error["status"] == "error"
        assert f"Parameter '{param}' must be a list of" in _text(error)

    @pytest.mark.parametrize("param", sorted(_VECTOR_PARAM_OWNERS))
    def test_a_correctly_sized_numpy_vector_is_still_accepted(self, param: str) -> None:
        """Over-reach control: numpy vectors of the right width keep dispatching."""
        action, extra = _VECTOR_PARAM_OWNERS[param]
        width = MuJoCoSimEngine._VECTOR_PARAM_LENGTHS[param][0]
        engine = MuJoCoSimEngine(tool_name="unsized_router_ok", mesh=False)
        signature = inspect.signature(getattr(MuJoCoSimEngine, action))
        _, error = engine._validate_and_build_kwargs(
            action, action, signature, {**extra, param: np.linspace(0.1, 0.4, width)}
        )
        assert error is None, _text(error or {})


# --------------------------------------------------------------------------- #
# get_world_point: pixels, and each [u, v] pair.
# --------------------------------------------------------------------------- #
class TestGetWorldPointPixels:
    """Both length probes on the pixel list keep the never-raises envelope.

    These are the structural checks the method runs before any render work, so
    they are reachable without a compiled world.
    """

    def test_an_unsized_pixels_container_is_reported(self) -> None:
        """A 0-d array in place of the pixel list is refused with the usage hint."""
        engine = MuJoCoSimEngine(tool_name="unsized_pixels", mesh=False)
        result = _call(engine.get_world_point, pixels=np.array(320), camera_name="cam")
        assert result["status"] == "error"
        assert "get_world_point requires 'pixels'" in _text(result)

    def test_an_unsized_pixel_pair_is_reported(self) -> None:
        """A 0-d array in place of one [u, v] pair names that pair's index."""
        engine = MuJoCoSimEngine(tool_name="unsized_pixel_pair", mesh=False)
        result = _call(engine.get_world_point, pixels=[np.array(320)], camera_name="cam")
        assert result["status"] == "error"
        assert "pixels[0] must be a [u, v] pair" in _text(result)

    def test_a_numpy_pixel_array_still_passes_structural_validation(self) -> None:
        """Over-reach control: a correctly shaped numpy pixel array reaches the render.

        With no world compiled the render is what fails, which is exactly the
        evidence that the structural checks accepted the pixels.
        """
        engine = MuJoCoSimEngine(tool_name="sized_pixels", mesh=False)
        result = _call(engine.get_world_point, pixels=np.array([[320, 240]]), camera_name="cam")
        assert result["status"] == "error"
        assert "failed to render camera frame" in _text(result)


# --------------------------------------------------------------------------- #
# send_action's ordered-vector form.
# --------------------------------------------------------------------------- #
class TestSendActionVectorForm:
    """An unsized action reports the mapping-or-vector contract it violated."""

    def test_an_unsized_action_names_the_two_accepted_shapes(self, tmp_path: Path) -> None:
        """The caller is told what an action may be, not that a length failed."""
        model = tmp_path / "arm.xml"
        model.write_text(_ARM_MJCF)
        engine = MuJoCoSimEngine(tool_name="unsized_action", mesh=False)
        engine.create_world()
        engine.add_robot(name="arm", urdf_path=str(model))
        result = _call(engine.send_action, action=np.array(0.5))
        assert result["status"] == "error"
        message = _text(result)
        assert "must be a mapping" in message
        assert "ordered numeric" in message


# --------------------------------------------------------------------------- #
# mjpeg_frames: the documented ValueError, not a bare TypeError.
# --------------------------------------------------------------------------- #
class TestMjpegFrameSize:
    """``size`` fails through the documented ``Raises: ValueError`` channel."""

    def _frame(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def test_an_unsized_size_raises_the_documented_valueerror(self) -> None:
        """A 0-d array is refused by the eager validator, message included."""
        with pytest.raises(ValueError, match="size must be a .width, height. pair"):
            _call(mjpeg_frames, frame_fn=self._frame, size=np.array(640), max_frames=1)

    def test_a_numpy_size_pair_is_still_accepted(self) -> None:
        """Over-reach control: a 2-element numpy size keeps working."""
        stream = _call(mjpeg_frames, frame_fn=self._frame, size=np.array([640, 480]), max_frames=1)
        assert next(iter(stream)).startswith(b"--frame")


# --------------------------------------------------------------------------- #
# Structural: no envelope module may probe a caller length with hasattr again.
# --------------------------------------------------------------------------- #
_ENVELOPE_PACKAGES = ("simulation", "rendering")


def _hasattr_len_probes(tree: ast.AST) -> list[int]:
    """Line numbers of every ``hasattr(<x>, "__len__")`` call in ``tree``."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", "") != "hasattr":
            continue
        if len(node.args) != 2 or not isinstance(node.args[1], ast.Constant):
            continue
        if node.args[1].value == "__len__":
            lines.append(node.lineno)
    return lines


def _envelope_modules() -> list[Path]:
    """Every module of the packages that publish a no-raise envelope."""
    package_dir = Path(inspect.getfile(sequence_length)).parent
    modules: list[Path] = []
    for name in _ENVELOPE_PACKAGES:
        modules.extend(sorted((package_dir / name).rglob("*.py")))
    return modules


class TestNoDirectLengthProbe:
    """The shared owner cannot be bypassed by reintroducing the unsafe idiom."""

    def test_the_scan_root_resolves_to_real_modules(self) -> None:
        """Non-vacuity: a mislocated root would make the scan below pass trivially."""
        modules = _envelope_modules()
        assert len(modules) > 20, modules
        assert any(path.name == "base.py" for path in modules)

    def test_no_envelope_module_probes_a_length_with_hasattr(self) -> None:
        """``hasattr(x, "__len__")`` is never a safe stand-in for a length probe.

        A 0-d array passes it and then raises from ``len()``. Ask
        :func:`strands_robots.utils.sequence_length` for the length instead and
        branch on ``None``; use ``__getitem__`` when indexability is the question.
        """
        offenders = {
            f"{path.name}:{line}"
            for path in _envelope_modules()
            for line in _hasattr_len_probes(ast.parse(path.read_text()))
        }
        assert not offenders, f"use sequence_length() instead of a hasattr length probe: {sorted(offenders)}"

    def test_the_scanner_detects_a_planted_probe(self) -> None:
        """Meta: an empty result means clean sources, not a scanner matching nothing."""
        planted = ast.parse('if not hasattr(value, "__len__"):\n    pass\n')
        assert _hasattr_len_probes(planted) == [1]
