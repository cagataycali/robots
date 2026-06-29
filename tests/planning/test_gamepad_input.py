"""Behavior tests for :class:`GamepadInput` (pygame planner input source).

``GamepadInput`` maps an analog gamepad onto locomotion intent. These tests
inject a fake ``pygame`` module so the axis/button -> :class:`PlannerUpdate`
mapping, deadzone handling, device-open guard, and teardown are exercised on any
platform without a real gamepad or the optional ``pygame`` dependency installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import strands_robots.utils as sr_utils
from strands_robots.planning.base import STYLES


class _FakeJoystick:
    """Minimal pygame Joystick stub returning canned axis/button values."""

    def __init__(self, axes: list[float], buttons: list[int]) -> None:
        self._axes = axes
        self._buttons = buttons
        self.inited = False
        self.quit_called = False
        self.quit_raises = False

    def init(self) -> None:
        self.inited = True

    def get_axis(self, index: int) -> float:
        return self._axes[index]

    def get_numbuttons(self) -> int:
        return len(self._buttons)

    def get_button(self, index: int) -> int:
        return self._buttons[index]

    def quit(self) -> None:
        if self.quit_raises:
            raise RuntimeError("device teardown failed")
        self.quit_called = True


def _make_fake_pygame(joystick: _FakeJoystick | None, *, count: int = 1) -> Any:
    """Build a fake ``pygame`` module exposing only the surface GamepadInput uses."""
    joy_mod = types.SimpleNamespace(
        _inited=False,
        init=lambda: None,
        get_count=lambda: count,
        Joystick=lambda index: joystick,
    )
    fake = types.ModuleType("pygame")
    fake._inited = False  # type: ignore[attr-defined]
    fake.init = lambda: setattr(fake, "_inited", True)  # type: ignore[attr-defined]
    fake.joystick = joy_mod  # type: ignore[attr-defined]
    fake.event = types.SimpleNamespace(pump=lambda: None)  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def fake_pygame(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Inject a fake pygame and a one-button joystick; yield a factory.

    Clears the ``require_optional`` import cache around the test so the fake is
    resolved rather than any real ``pygame`` that may be installed.
    """

    def install(joystick: _FakeJoystick | None, *, count: int = 1) -> Any:
        fake = _make_fake_pygame(joystick, count=count)
        monkeypatch.setitem(sys.modules, "pygame", fake)
        sr_utils._lazy_modules.pop("pygame", None)
        return fake

    monkeypatch.setattr(sr_utils, "_lazy_modules", dict(sr_utils._lazy_modules))
    yield install


def _gamepad(**kwargs: Any):  # type: ignore[no-untyped-def]
    from strands_robots.planning.inputs import GamepadInput

    return GamepadInput(**kwargs)


def test_poll_before_start_returns_none(fake_pygame: Any) -> None:
    fake_pygame(_FakeJoystick(axes=[0.0, 0.0, 0.0, 0.0, 0.0], buttons=[0]))
    pad = _gamepad()
    assert pad.poll() is None


def test_start_raises_when_no_device_at_index(fake_pygame: Any) -> None:
    fake_pygame(None, count=0)
    pad = _gamepad()
    with pytest.raises(RuntimeError, match="no gamepad at index 0"):
        pad.start()


def test_axes_map_to_intent_with_inversion_and_scaling(fake_pygame: Any) -> None:
    # axis(1)=forward (inverted), axis(0)=lateral (inverted), axis(3)=yaw (inverted),
    # axis(4)=height trim. Pick values above the deadzone.
    joy = _FakeJoystick(axes=[-0.5, -1.0, 0.0, 0.5, 1.0], buttons=[0])
    fake_pygame(joy)
    pad = _gamepad(max_speed=2.0, max_omega=4.0, height_center=0.74, height_span=0.2)
    pad.start()
    assert joy.inited is True

    update = pad.poll()
    assert update is not None
    vx, vy, omega = update.root_vel  # type: ignore[misc]
    # vx = -axis(1)*max_speed = -(-1.0)*2.0 = 2.0
    assert vx == pytest.approx(2.0)
    # vy = -axis(0)*max_speed = -(-0.5)*2.0 = 1.0
    assert vy == pytest.approx(1.0)
    # omega = -axis(3)*max_omega = -(0.5)*4.0 = -2.0
    assert omega == pytest.approx(-2.0)
    # height = center - axis(4)*span = 0.74 - 1.0*0.2 = 0.54
    assert update.height == pytest.approx(0.54)
    assert update.style is None


def test_deadzone_zeroes_small_axis_values(fake_pygame: Any) -> None:
    joy = _FakeJoystick(axes=[0.05, -0.05, 0.09, 0.0, 0.0], buttons=[0])
    fake_pygame(joy)
    pad = _gamepad(deadzone=0.1, height_center=0.74, height_span=0.2)
    pad.start()
    update = pad.poll()
    assert update is not None
    assert update.root_vel == (0.0, 0.0, 0.0)
    # axis(4)=0 within deadzone -> height stays at center.
    assert update.height == pytest.approx(0.74)


def test_first_pressed_button_selects_style(fake_pygame: Any) -> None:
    # Press the button whose index maps to STYLES[2] == "stealth"; ensure the
    # first pressed button wins (button 2 set, button 5 also set).
    buttons = [0, 0, 1, 0, 0, 1]
    joy = _FakeJoystick(axes=[0.0, 0.0, 0.0, 0.0, 0.0], buttons=buttons)
    fake_pygame(joy)
    pad = _gamepad()
    pad.start()
    update = pad.poll()
    assert update is not None
    assert update.style == STYLES[2]


def test_start_is_idempotent(fake_pygame: Any) -> None:
    joy = _FakeJoystick(axes=[0.0, 0.0, 0.0, 0.0, 0.0], buttons=[0])
    fake = fake_pygame(joy)
    pad = _gamepad()
    pad.start()
    fake._inited = False  # type: ignore[attr-defined]
    pad.start()  # joystick already open -> early return, no re-init
    assert fake._inited is False  # type: ignore[attr-defined]


def test_stop_quits_joystick_and_swallows_teardown_error(fake_pygame: Any) -> None:
    joy = _FakeJoystick(axes=[0.0, 0.0, 0.0, 0.0, 0.0], buttons=[0])
    joy.quit_raises = True
    fake_pygame(joy)
    pad = _gamepad()
    pad.start()
    pad.stop()  # quit() raises internally; must be swallowed
    assert pad.poll() is None  # joystick cleared, poll is a no-op again


def test_reset_is_noop(fake_pygame: Any) -> None:
    joy = _FakeJoystick(axes=[0.0, 0.0, 0.0, 0.0, 0.0], buttons=[0])
    fake_pygame(joy)
    pad = _gamepad()
    pad.start()
    pad.reset()
    update = pad.poll()
    assert update is not None  # still operational after reset
