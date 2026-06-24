"""Behavior tests for the ``use_lerobot`` universal LeRobot access tool.

``use_lerobot`` is to ``lerobot`` what ``use_aws`` is to boto3: a single
dispatcher that resolves any dotted path into the lerobot package and either
describes or calls it, with config choices discovered dynamically from
lerobot's own draccus ``ChoiceRegistry`` registries (never hardcoded).

These tests pin the contracts that make the tool trustworthy:

1. Every user-facing ``text`` field is plain ASCII (the project's no-emoji rule).
2. Import resolution distinguishes three failure modes precisely:
   genuinely-missing paths, paths that exist but need an optional dependency,
   and attribute-not-found on a resolved object.
3. The serializer is total -- it never raises on circular references, runaway
   nesting, bytes, numpy scalars/arrays, or objects with a hostile ``__repr__``.
4. Introspection never triggers descriptor/property side effects.
5. Image arrays become real Strands ``image`` content blocks with a sane codec.

All tests are hardware-free and do not require the optional ``lerobot[dataset]``
extra; the missing-dependency path is asserted precisely *because* it is absent.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.use_lerobot as M

# The tool is wrapped by the Strands @tool decorator; call the raw function.
_fn = getattr(M.use_lerobot, "__wrapped__", None) or M.use_lerobot

pytest.importorskip("lerobot", reason="use_lerobot requires the lerobot package")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _texts(result: dict[str, Any]) -> str:
    """Concatenate all content ``text`` fields from a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _assert_ascii(text: str) -> None:
    """Fail if any character is outside the ASCII range."""
    offenders = {hex(ord(c)) for c in text if ord(c) > 127}
    assert not offenders, f"non-ASCII characters in tool output: {offenders}"


def _images(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [item["image"] for item in result.get("content", []) if "image" in item]


# ----------------------------------------------------------------------------
# discovery + registries
# ----------------------------------------------------------------------------
def test_discovery_lists_packages_and_registries() -> None:
    """Discovery enumerates packages and at least the four config registries."""
    result = _fn(module="__discovery__", method="list_modules")
    assert result["status"] == "success"
    text = _texts(result)
    _assert_ascii(text)
    assert "LeRobot API Discovery" in text
    for kind in ("robots", "teleoperators", "cameras", "policies"):
        assert kind in text


def test_registry_listing_is_dynamic_not_hardcoded() -> None:
    """A registry listing reflects lerobot's own registered choices."""
    result = _fn(module="__registry__", method="robots")
    assert result["status"] == "success"
    text = _texts(result)
    _assert_ascii(text)
    assert "registry" in text
    # so100/so101 are stable, long-lived SO-arm choices.
    assert "so100_follower" in text or "so101_follower" in text


def test_unknown_registry_reports_valid_kinds() -> None:
    result = _fn(module="__registry__", method="totally_not_a_registry")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "Valid:" in text
    assert "robots" in text and "policies" in text


def test_empty_registry_method_defaults_to_robots() -> None:
    """An empty registry method falls back to the robots registry."""
    result = _fn(module="__registry__", method="")
    assert result["status"] == "success"
    assert "robots" in _texts(result)


# ----------------------------------------------------------------------------
# import resolution -- the three failure modes
# ----------------------------------------------------------------------------
def test_genuinely_missing_path_is_cannot_resolve() -> None:
    """A path with no lerobot module behind it -> 'Cannot resolve' + a tip."""
    result = _fn(module="does.not.exist", method="foo")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "Cannot resolve" in text
    assert "__discovery__" in text  # actionable tip
    # Must NOT misdirect the user to install an optional extra.
    assert "lerobot[dataset]" not in text


def test_fake_lerobot_submodule_is_cannot_resolve() -> None:
    result = _fn(module="robots.totally_fake_robot", method="x")
    assert result["status"] == "error"
    assert "Cannot resolve" in _texts(result)


def test_missing_optional_dependency_surfaces_real_error() -> None:
    """A real path needing an absent extra surfaces the dependency, not a
    misleading 'cannot resolve'. ``datasets`` is intentionally not installed."""
    if importlib.util.find_spec("datasets") is not None:
        pytest.skip("datasets extra is installed; cannot assert the missing-dep path")
    result = _fn(module="datasets.lerobot_dataset.LeRobotDataset", method="__describe__")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "exists but failed to import" in text
    assert "lerobot[dataset]" in text  # the actionable extra


def test_attribute_not_found_lists_available() -> None:
    """A bad method on a resolved module lists real alternatives."""
    result = _fn(module="policies.factory", method="definitely_not_here")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "not found" in text
    assert "Available:" in text


# ----------------------------------------------------------------------------
# calling + signatures
# ----------------------------------------------------------------------------
def test_call_with_params_succeeds() -> None:
    result = _fn(
        module="policies.factory",
        method="get_policy_class",
        parameters={"name": "act"},
    )
    assert result["status"] == "success"
    _assert_ascii(_texts(result))


def test_missing_required_arg_reports_signature() -> None:
    """A TypeError from a bad call surfaces the expected signature."""
    result = _fn(module="policies.factory", method="get_policy_class")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "TypeError" in text
    assert "name" in text  # the missing parameter is named


def test_read_constant_attribute() -> None:
    result = _fn(module="utils.constants", method="HF_LEROBOT_CALIBRATION")
    assert result["status"] == "success"
    _assert_ascii(_texts(result))


# ----------------------------------------------------------------------------
# introspection -- describe without side effects
# ----------------------------------------------------------------------------
def test_describe_separates_properties_from_methods() -> None:
    """``__describe__`` classifies properties, class methods, and instance
    methods distinctly, using static lookup (no descriptor side effects)."""
    import json

    result = _fn(module="cameras.opencv.OpenCVCamera", method="__describe__")
    assert result["status"] == "success"
    text = _texts(result)
    _assert_ascii(text)
    info = json.loads(text.split("\n", 1)[1])
    # is_connected is a property on the camera class, not a callable method.
    assert "is_connected" in info.get("properties", [])
    assert "is_connected" not in info.get("methods", [])
    # find_cameras is a classmethod; it should not be double-listed as a method.
    assert "find_cameras" in info.get("class_methods", [])
    assert "find_cameras" not in info.get("methods", [])


# ----------------------------------------------------------------------------
# serializer -- totality under hostile input
# ----------------------------------------------------------------------------
def test_serializer_handles_circular_dict() -> None:
    d: dict[str, Any] = {}
    d["self"] = d
    out = M._serialize_result(d)
    assert "circular ref" in out


def test_serializer_handles_circular_list() -> None:
    lst: list[Any] = []
    lst.append(lst)
    assert "circular ref" in M._serialize_result(lst)


def test_serializer_handles_bytes_structurally() -> None:
    out = M._serialize_value(b"\x00\x01\x02hello")
    assert out["__bytes__"] is True
    assert out["length"] == 8
    assert out["preview_hex"].startswith("000102")


def test_serializer_handles_numpy_scalars() -> None:
    out = M._serialize_value({"i": np.int64(5), "f": np.float32(1.5)})
    assert out["i"] == 5
    assert out["f"] == pytest.approx(1.5)


def test_serializer_summarizes_large_arrays_structurally() -> None:
    out = M._serialize_value(np.zeros((2, 3, 64, 64)))
    assert isinstance(out, dict)
    assert out["__ndarray__"] is True
    assert out["shape"] == [2, 3, 64, 64]
    # A 4D tensor is not pixel-dumped as text.
    assert "values" not in out


def test_serializer_survives_hostile_repr() -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("boom")

    # Must not raise -- the tool stays alive even on pathological objects.
    out = M._serialize_result({"x": Hostile()})
    assert isinstance(out, str)


def test_serializer_depth_guard() -> None:
    """Deeply nested structures terminate rather than blowing the stack."""
    node: dict[str, Any] = {}
    cur = node
    for _ in range(200):
        nxt: dict[str, Any] = {}
        cur["next"] = nxt
        cur = nxt
    out = M._serialize_result(node)
    assert "max depth exceeded" in out


# ----------------------------------------------------------------------------
# image detection + encoding
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,expected",
    [
        ((100, 100, 3), True),  # RGB
        ((100, 100), True),  # grayscale
        ((100, 100, 4), True),  # RGBA
        ((1, 1, 3), False),  # too small
        ((10, 10, 2), False),  # invalid channel count
        ((2, 3, 64, 64), False),  # 4D tensor, not an image
    ],
)
def test_image_detection_heuristic(shape: tuple[int, ...], expected: bool) -> None:
    arr = np.zeros(shape, dtype=np.uint8)
    assert M._is_image_array(arr) is expected


def test_large_rgb_frame_encodes_as_jpeg() -> None:
    """Large opaque frames use JPEG (compact in-context)."""
    pytest.importorskip("cv2")
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    block = M._array_to_image_block(frame)
    assert block is not None
    assert block["image"]["format"] == "jpeg"
    assert block["image"]["source"]["bytes"]


def test_small_frame_and_alpha_use_png() -> None:
    """Small frames and anything with alpha stay PNG (lossless / alpha-safe)."""
    pytest.importorskip("cv2")
    small = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    rgba = (np.random.rand(480, 640, 4) * 255).astype(np.uint8)
    small_block = M._array_to_image_block(small)
    rgba_block = M._array_to_image_block(rgba)
    assert small_block is not None
    assert rgba_block is not None
    assert small_block["image"]["format"] == "png"
    assert rgba_block["image"]["format"] == "png"


def test_collect_images_finds_frames_in_dict() -> None:
    """Images nested one level inside a dict (camera_name -> frame) are found."""
    pytest.importorskip("cv2")
    frames = {
        "front": (np.random.rand(64, 64, 3) * 255).astype(np.uint8),
        "wrist": (np.random.rand(64, 64, 3) * 255).astype(np.uint8),
        "meta": "not an image",
    }
    blocks: list[dict[str, Any]] = []
    M._collect_images(frames, blocks)
    assert len(blocks) == 2


# ----------------------------------------------------------------------------
# deep-import cache
# ----------------------------------------------------------------------------
def test_deep_import_is_cached() -> None:
    """A second deep import of the same package is a no-op (cache hit)."""
    M._DEEP_IMPORTED.discard("lerobot.policies")
    M._deep_import("lerobot.policies")
    assert "lerobot.policies" in M._DEEP_IMPORTED
    # Idempotent: calling again must not raise and the marker persists.
    M._deep_import("lerobot.policies")
    assert "lerobot.policies" in M._DEEP_IMPORTED
