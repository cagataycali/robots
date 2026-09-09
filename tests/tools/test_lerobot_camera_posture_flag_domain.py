"""A camera posture flag is checked, never read by truthiness.

``lerobot_camera`` tables its numeric options and its two vocabularies, and read
its three posture flags - ``async_mode``, ``warmup`` and ``save_config`` - raw.
Every non-empty string is truthy, so the words a caller reaches for when opting
out selected the affirmative posture. Measured on ``eecaa80``:

* ``save_config="false"`` wrote the configuration file under
  ``status="success"``, leaving a durable artifact for a caller who spelled the
  opt-out;
* ``warmup="false"`` was passed to ``Camera.connect`` as the string, persisted
  into that file as ``"warmup": "false"`` - the one field of that document
  declared a boolean - and reported on the line above it as ``Warmup: on``;
* ``async_mode="false"`` selected the asynchronous read path, which the plain
  ``False`` does not.

The flag also *gates* a tabled numeric row: ``timeout_ms`` is only refused under
``async_mode``, because the synchronous read consumes no budget. Reading that
gate by truthiness switched the row off from outside its own table -
``async_mode=0`` with ``timeout_ms=-5`` was answered ``status="success"``.

These tests pin that each flag an action consumes is refused unless it is a
boolean, that the refusal happens before the effect the truthy value had, that
the budget row cannot be gated off by a value that is not a declared spelling of
*off*, that an action consuming none of the three still refuses none of them, and
that a flag added to the signature cannot skip the roster.
"""

from __future__ import annotations

import glob
import inspect
import json
import os
import time
from typing import Any

import numpy as np
import pytest

# The module object is needed for the ``cv2``/``os`` handles the tool imports and
# for the roster's signature scan, so every name is reached through this alias.
import strands_robots.tools.lerobot_camera as cam_mod
from strands_robots.utils import boolean_flag_error

# One value per rejection reason of the shared posture domain: the two spellings
# of *off* that read as *on*, the integers that pass as a silent posture, and the
# two values that take a branch without being a declared spelling of it.
BAD_POSTURES: tuple[Any, ...] = ("false", "no", 0, 1, None, [])

# The flag each action is actually handed, which is what may be refused.
ACTION_FLAGS = tuple((action, flag) for action, flags in cam_mod._ACTION_POSTURE_FLAGS.items() for flag in flags)


class _Camera:
    """A camera stand-in recording every posture and read path it is driven with."""

    def __init__(self) -> None:
        self.warmups: list[Any] = []
        self.sync_reads = 0
        self.async_reads = 0
        self.width = 8
        self.height = 6
        self.fps = 30
        self.color_mode = type("_M", (), {"value": "RGB"})()
        self.rotation: Any = None

    def connect(self, warmup: bool = True) -> None:
        self.warmups.append(warmup)

    def disconnect(self) -> None:
        return None

    def _frame(self) -> np.ndarray:
        # A measurable span keeps the tool's own rate arithmetic off zero.
        time.sleep(0.0005)
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def read(self) -> np.ndarray:
        self.sync_reads += 1
        return self._frame()

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        self.async_reads += 1
        return self._frame()


@pytest.fixture
def camera(monkeypatch: pytest.MonkeyPatch) -> _Camera:
    """Substitute the camera factory and neutralise every sink a handler writes to."""
    cam = _Camera()
    monkeypatch.setattr(cam_mod, "_create_camera", lambda *a, **k: cam)
    writer = type("_W", (), {"write": lambda self, f: None, "release": lambda self: None})()
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cam_mod.cv2, "imwrite", lambda *a, **k: True)
    monkeypatch.setattr(cam_mod.cv2, "imshow", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cam_mod.cv2, "waitKey", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cam_mod.cv2, "destroyAllWindows", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 1234)
    return cam


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with agent-shaped values.

    The three flags are annotated ``bool`` and every value under test here is
    deliberately outside that annotation, so one ``**kwargs: Any`` helper states
    that once rather than scattering a suppression over every call site.
    """
    return cam_mod.lerobot_camera(**kwargs)


def _action_kwargs(action: str, tmp_path: Any) -> dict[str, Any]:
    """The smallest set of options that drives ``action`` through one read."""
    common: dict[str, Any] = {
        "action": action,
        "camera_type": "opencv",
        "save_path": str(tmp_path),
        "width": 8,
        "height": 6,
        "fps": 2,
    }
    if action == "capture_batch":
        common["camera_ids"] = [0]
    else:
        common["camera_id"] = 0
    if action == "record":
        common["capture_duration"] = 0.5
    if action == "preview":
        common["preview_duration"] = 0.01
    return common


def _saved_configs(tmp_path: Any) -> list[str]:
    return glob.glob(os.path.join(str(tmp_path), "camera_config_*.json"))


class TestAPostureTheToolCannotReadIsRefused:
    """Every flag an action consumes is held to the shared boolean domain."""

    @pytest.mark.parametrize(("action", "flag"), ACTION_FLAGS)
    @pytest.mark.parametrize("value", BAD_POSTURES)
    def test_a_non_boolean_posture_is_refused_before_the_camera_opens(
        self, camera: _Camera, tmp_path: Any, action: str, flag: str, value: Any
    ) -> None:
        result = _call(**_action_kwargs(action, tmp_path) | {flag: value})

        assert result["status"] == "error"
        assert _text(result).startswith(f"{action}: {flag} must be a boolean")
        assert camera.warmups == [], "the camera was opened on a posture the tool cannot read"

    @pytest.mark.parametrize("value", BAD_POSTURES + (True, False, np.True_, np.False_))
    def test_the_posture_domain_matches_the_shared_helper(self, camera: _Camera, tmp_path: Any, value: Any) -> None:
        refused = _call(**_action_kwargs("capture", tmp_path) | {"async_mode": value})["status"] == "error"

        assert refused == (boolean_flag_error(value, "async_mode", "capture") is not None)

    def test_an_action_that_consumes_no_posture_refuses_none_of_them(self, camera: _Camera, tmp_path: Any) -> None:
        """``discover`` reads none of the three, so a value it never consults stands.

        The rule the numeric table already follows: an option no handler consumes
        must not be refused.
        """
        result = _call(action="discover", camera_type="opencv", warmup="false", save_config="false")

        assert "must be a boolean" not in _text(result)


class TestTheRefusalPrecedesTheEffectTheTruthyValueHad:
    """The three postures measured on the pre-fix tree, each now not taken."""

    def test_an_unreadable_save_posture_writes_no_configuration_file(self, camera: _Camera, tmp_path: Any) -> None:
        result = _call(**_action_kwargs("configure", tmp_path) | {"save_config": "false"})

        assert result["status"] == "error"
        assert _saved_configs(tmp_path) == [], "the opt-out left a configuration file behind"

    def test_an_unreadable_warmup_posture_reaches_neither_the_driver_nor_the_file(
        self, camera: _Camera, tmp_path: Any
    ) -> None:
        result = _call(**_action_kwargs("configure", tmp_path) | {"save_config": True, "warmup": "false"})

        assert result["status"] == "error"
        assert camera.warmups == []
        assert _saved_configs(tmp_path) == []

    def test_an_unreadable_read_path_posture_performs_no_asynchronous_read(
        self, camera: _Camera, tmp_path: Any
    ) -> None:
        result = _call(**_action_kwargs("capture", tmp_path) | {"async_mode": "false"})

        assert result["status"] == "error"
        assert (camera.async_reads, camera.sync_reads) == (0, 0)


class TestAnHonestPostureStillSelectsBothPaths:
    """The refusal must not cost a caller who supplies the declared domain."""

    @pytest.mark.parametrize(
        ("async_mode", "expected"),
        [(True, (1, 0)), (False, (0, 1)), (np.True_, (1, 0)), (np.False_, (0, 1))],
        ids=["true", "false", "numpy-true", "numpy-false"],
    )
    def test_a_boolean_read_path_is_honoured_unchanged(
        self, camera: _Camera, tmp_path: Any, async_mode: Any, expected: tuple[int, int]
    ) -> None:
        result = _call(**_action_kwargs("capture", tmp_path) | {"async_mode": async_mode})

        assert result["status"] == "success"
        assert (camera.async_reads, camera.sync_reads) == expected

    def test_a_boolean_save_posture_still_persists_the_posture_it_was_given(
        self, camera: _Camera, tmp_path: Any
    ) -> None:
        result = _call(**_action_kwargs("configure", tmp_path) | {"save_config": True, "warmup": False})

        assert result["status"] == "success"
        saved = _saved_configs(tmp_path)
        assert len(saved) == 1
        with open(saved[0], encoding="utf-8") as handle:
            persisted = json.load(handle)
        assert persisted["warmup"] is False
        assert "Warmup: off" in _text(result)
        assert camera.warmups == [False]


class TestTheGateCannotSwitchOffTheBudgetRow:
    """``timeout_ms`` is refused under ``async_mode``, so the gate is graded first."""

    def test_a_falsy_gate_no_longer_discards_the_budget_row(self, camera: _Camera, tmp_path: Any) -> None:
        """Pre-fix ``async_mode=0`` reported success with an unusable budget."""
        result = _call(**_action_kwargs("capture", tmp_path) | {"async_mode": 0, "timeout_ms": -5})

        assert result["status"] == "error"
        assert camera.sync_reads == 0

    def test_the_refusal_names_the_flag_rather_than_the_budget_it_gates(self, camera: _Camera, tmp_path: Any) -> None:
        """Placement pin: the numeric guard would blame ``timeout_ms`` instead."""
        result = _call(**_action_kwargs("capture", tmp_path) | {"async_mode": "false", "timeout_ms": -5})

        assert _text(result).startswith("capture: async_mode must be a boolean")

    def test_a_boolean_gate_still_decides_whether_the_budget_is_read(self, camera: _Camera, tmp_path: Any) -> None:
        """The row is still gated, so a budget the synchronous read never uses stands."""
        result = _call(**_action_kwargs("capture", tmp_path) | {"async_mode": False, "timeout_ms": -5})

        assert result["status"] == "success"
        assert camera.sync_reads == 1


class TestThePostureRosterFollowsTheSignature:
    """Structural pins: a flag cannot be added to the tool and skip the roster."""

    def test_every_boolean_parameter_of_the_tool_is_on_the_roster(self) -> None:
        annotations = inspect.signature(cam_mod.lerobot_camera).parameters
        declared = {name for name, param in annotations.items() if param.annotation in (bool, "bool")}
        rostered = {flag for flags in cam_mod._ACTION_POSTURE_FLAGS.values() for flag in flags}

        assert declared == rostered, f"posture flag read without the domain: {declared - rostered}"

    def test_the_roster_covers_every_action_that_opens_a_camera(self) -> None:
        """The same six actions the numeric table covers; the other two open none."""
        assert set(cam_mod._ACTION_POSTURE_FLAGS) == set(cam_mod._ACTION_NUMERIC_OPTIONS)
