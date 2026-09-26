"""One stand-in for the camera ``lerobot_camera`` opens, bound to its real contract.

Every test that drives :mod:`strands_robots.tools.lerobot_camera` replaces
``_create_camera`` -- the one seam between the tool and a physical device -- and
each of them used to hand back its own hand-rolled object. Eight such doubles
existed across eight files, all modelling the same four-method surface, and none
of them was bound to it: a ``connect``/``read``/``async_read`` spelled with a
keyword no lerobot camera declares was accepted by all eight, and the factory
itself was replaced by ``lambda *a, **k: cam``, which accepts any arity at all.

Nothing else covers that. lerobot ships no ``py.typed``, so every camera symbol
this tool imports is ``Any`` to mypy: how the tool calls a camera is pinned by
these tests or by nothing.

So the stand-in here takes its shape FROM the symbols the module under test
holds. Each recorded call is bound against the signature of the method it
replaces on every camera the factory can return -- the declared ``Camera``
contract and both concrete backends -- and the factory replacement is bound
against ``_create_camera``'s own signature. A call an lerobot camera would
refuse is therefore refused here too, and the geometry a handler opened is read
off the factory's parameter names rather than a positional slice.

Usage::

    @pytest.fixture
    def camera(monkeypatch: pytest.MonkeyPatch) -> Camera:
        return stands_in_for(monkeypatch, read_seconds=0.0005)
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_camera as cam_mod

#: Every camera ``_create_camera`` can hand back, read off the module under test
#: so this tuple cannot name a class the tool does not use: the ``Camera``
#: contract it declares as its return type, plus both concrete backends. A call
#: must satisfy all of them, because a handler does not know which one it holds.
#: ``RealSenseCamera`` is ``None`` when lerobot's RealSense modules are absent.
CONTRACTS = tuple(
    contract for contract in (cam_mod.Camera, cam_mod.OpenCVCamera, cam_mod.RealSenseCamera) if contract is not None
)


@dataclass(frozen=True)
class Opened:
    """One ``_create_camera`` call, read through the factory's parameter names."""

    camera_type: Any
    camera_id: Any
    width: Any
    height: Any
    fps: Any
    color_mode: Any
    rotation: Any


class Camera:
    """A camera stand-in that records what it was driven with, and only that.

    Args:
        width: Frame width the device reports, and the width of the frame served.
        height: Frame height the device reports, and of the frame served.
        fps: Rate the device reports.
        color_mode: Value behind the ``color_mode.value`` a handler reports.
        rotation: The ``rotation`` attribute a handler reports when it is not None.
        read_seconds: Real seconds each read takes. A measurable span keeps the
            tool's own rate arithmetic off zero; a span wide enough to place a
            clock step inside models a slow device.
        connect_seconds: Real seconds ``connect`` takes.
        needs_ms: Models a device slower than the budget it is handed: an
            asynchronous read whose budget does not cover it raises the driver's
            own ``TimeoutError``, naming the budget, the way
            :meth:`lerobot.cameras.opencv.OpenCVCamera.async_read` does.
        connect_error: Raised by ``connect``, for the device that vanishes
            between enumeration and use (cable yanked, busy handle, driver fault).
        frame: The frame both reads serve. Defaults to a black frame of the
            declared geometry; a cell that measures saved pixels supplies its own.
    """

    def __init__(
        self,
        *,
        width: int = 8,
        height: int = 6,
        fps: int = 30,
        color_mode: str = "RGB",
        rotation: Any = None,
        read_seconds: float = 0.0,
        connect_seconds: float = 0.0,
        needs_ms: float = 0.0,
        connect_error: BaseException | None = None,
        frame: np.ndarray | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.color_mode = SimpleNamespace(value=color_mode)
        self.rotation = rotation
        self.read_seconds = read_seconds
        self.connect_seconds = connect_seconds
        self.needs_ms = needs_ms
        self.connect_error = connect_error
        self.frame = frame
        #: Called with each :class:`Opened` row, for a device whose behaviour is
        #: decided by the configuration it was opened with.
        self.configure: Callable[[Opened], None] | None = None
        #: Every camera the factory was asked to open, in order.
        self.opened: list[Opened] = []
        #: The warmup posture each ``connect`` was driven with.
        self.warmups: list[Any] = []
        #: The budget each asynchronous read was handed.
        self.budgets: list[Any] = []
        self.sync_reads = 0
        self.async_reads = 0
        self.disconnect_calls = 0
        self.connected = False

    def _bind(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> inspect.BoundArguments:
        """Bind one call against every camera contract, and report it as read.

        Raises:
            TypeError: When any camera the factory can return would refuse the
                call -- an argument it has no parameter for, or a required one
                left out.
        """
        bound: inspect.BoundArguments | None = None
        for contract in CONTRACTS:
            try:
                candidate = inspect.signature(getattr(contract, method)).bind(self, *args, **kwargs)
            except TypeError as exc:
                raise TypeError(f"{contract.__name__}.{method} would refuse this call: {exc}") from exc
            bound = bound if bound is not None else candidate
        assert bound is not None, "no camera contract was resolved"
        return bound

    def connect(self, *args: Any, **kwargs: Any) -> None:
        bound = self._bind("connect", args, kwargs)
        self.warmups.append(bound.arguments.get("warmup", True))
        if self.connect_seconds:
            time.sleep(self.connect_seconds)
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    def disconnect(self, *args: Any, **kwargs: Any) -> None:
        self._bind("disconnect", args, kwargs)
        self.disconnect_calls += 1
        self.connected = False

    def read(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self._bind("read", args, kwargs)
        self.sync_reads += 1
        return self._frame()

    def async_read(self, *args: Any, **kwargs: Any) -> np.ndarray:
        bound = self._bind("async_read", args, kwargs)
        budget = bound.arguments.get("timeout_ms")
        self.budgets.append(budget)
        self.async_reads += 1
        if budget is not None and self.needs_ms > float(budget):
            raise TimeoutError(f"Timed out waiting for frame from camera after {budget} ms.")
        return self._frame()

    def _frame(self) -> np.ndarray:
        if self.read_seconds:
            time.sleep(self.read_seconds)
        if self.frame is not None:
            return self.frame
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


def stands_in_for(monkeypatch: pytest.MonkeyPatch, **options: Any) -> Camera:
    """Put one stand-in behind ``_create_camera`` and record every call to it.

    Args:
        monkeypatch: The cell's patcher; the seam is restored with it.
        **options: Forwarded to :class:`Camera`.

    Returns:
        The stand-in every handler will be handed.
    """
    camera = Camera(**options)
    signature = inspect.signature(cam_mod._create_camera)

    def _create(*args: Any, **kwargs: Any) -> Camera:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        opened = Opened(**bound.arguments)
        camera.opened.append(opened)
        if camera.configure is not None:
            camera.configure(opened)
        return camera

    monkeypatch.setattr(cam_mod, "_create_camera", _create)
    return camera
