"""``REALSENSE_AVAILABLE`` answers whether a RealSense camera can be opened.

lerobot imports its RealSense camera classes whether or not the Intel SDK is
installed: ``pyrealsense2`` is bound to ``None`` behind an availability flag and
required at the call sites instead. So that import succeeding says the classes
exist, not that the SDK is there - which is the question every use of the flag
asks. Deriving the flag from the import alone reported ``SDK Available:  Yes``
and ``Depth Support:  Yes`` for a host with no SDK, left the branch that names
the install unreachable, and sent the creator to build a camera lerobot then
refused.

These pin the flag against the SDK itself, and pin that the two surfaces which
report the SDK absent - the details report and the camera creator - name one
install between them. A host that does have the SDK still exercises the same
equality, from the other side.
"""

from __future__ import annotations

import importlib.util

import pytest

import strands_robots.tools.lerobot_camera as cam_mod
from strands_robots.tools.lerobot_camera import lerobot_camera

# The install that works on every platform this package supports: on macOS the
# wheel ships as ``pyrealsense2-macosx``, which a bare ``pip install
# pyrealsense2`` does not reach.
_EXTRA_INSTALL = "pip install 'lerobot[intelrealsense]'"


def _sdk_present() -> bool:
    """True when the RealSense SDK is importable under its import name."""
    return importlib.util.find_spec("pyrealsense2") is not None


def _texts(result: dict) -> str:
    """Join every text block of a tool result."""
    return "\n".join(block.get("text", "") for block in result["content"])


def test_the_flag_reports_the_sdk_and_not_the_lerobot_import() -> None:
    """The flag equals the SDK's presence, whichever way round that is."""
    assert cam_mod.REALSENSE_AVAILABLE is _sdk_present()


def test_lerobot_imports_its_realsense_classes_with_the_sdk_absent() -> None:
    """The premise: the import cannot decide the SDK, so the flag cannot use it.

    Should lerobot go back to failing that import, the flag stays correct - it
    reads the SDK - and this cell is what says the premise moved.
    """
    if _sdk_present():
        pytest.skip("the SDK is installed, so the import decides nothing here")
    assert importlib.util.find_spec("lerobot.cameras.realsense.camera_realsense") is not None
    assert cam_mod.RealSenseCamera is not None


def test_creating_a_realsense_camera_without_the_sdk_reports_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent SDK is refused as an absent SDK, not as an unsupported type.

    "realsense" is a supported camera type on every platform, so answering the
    request with ``Unsupported camera type: realsense`` sends the caller looking
    for a spelling that does not exist instead of at the install they need. It is
    still a refusal either way - no camera is returned, and nothing falls back to
    OpenCV behind the caller's back.
    """
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", False)

    with pytest.raises(ImportError) as excinfo:
        cam_mod._create_camera("realsense", "0", 640, 480, 30, "RGB", "NO_ROTATION")

    assert excinfo.value.name == "pyrealsense2"
    assert _EXTRA_INSTALL in str(excinfo.value)
    assert "Unsupported camera type" not in str(excinfo.value)


def test_an_unsupported_camera_type_is_still_refused_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: a type this tool really does not support keeps its refusal."""
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", False)

    with pytest.raises(ValueError, match="Unsupported camera type: thermal"):
        cam_mod._create_camera("thermal", "0", 640, 480, 30, "RGB", "NO_ROTATION")


def test_both_surfaces_report_the_absent_sdk_with_one_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The details report and the creator name the same install, from one owner."""
    monkeypatch.setattr(cam_mod, "REALSENSE_AVAILABLE", False)

    assert _EXTRA_INSTALL in cam_mod.REALSENSE_SDK_ABSENT

    details = _texts(lerobot_camera(action="list", camera_type="realsense"))
    assert cam_mod.REALSENSE_SDK_ABSENT in details

    with pytest.raises(ImportError) as excinfo:
        cam_mod._create_camera("realsense", "0", 640, 480, 30, "RGB", "NO_ROTATION")
    assert str(excinfo.value) == cam_mod.REALSENSE_SDK_ABSENT
