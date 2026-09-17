"""``render_video.py --camera default`` collects pixels, not tool envelopes.

The example prefers a direct ``mujoco.Renderer`` with a body-tracking camera and
falls back to the engine's free camera - the path ``--camera default`` selects
outright, and the one every machine without an offscreen GL context takes. That
fallback read frames from ``sim.render()``, which answers the agent-tool PNG
envelope (``{"status", "content": [...]}``), so ``frames`` filled with dicts and
the first/last spread read died on the whole run::

    TypeError: int() argument must be a string, a bytes-like object or a real
    number, not 'dict'

``sim.get_frame()`` is the raw-pixel counterpart the engine documents for
in-process consumers. These cells pin that the fallback reads it, that the
frames it yields survive the spread arithmetic that crashed, and that the
envelope accessor is not consulted - each on a fake sim, so no GL, assets or
ONNX are needed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "microduck" / "render_video.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_video_frames_under_test", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeSim:
    """A sim whose pixels come from ``get_frame``; ``render`` is a trap."""

    def __init__(self, fill: int = 7) -> None:
        self.fill = fill
        self.calls: list[tuple[str, int, int]] = []

    def get_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        assert width is not None and height is not None
        self.calls.append((camera_name, width, height))
        rgb = np.full((height, width, 3), self.fill, dtype=np.uint8)
        return rgb, np.zeros((height, width), dtype=np.float32)

    def render(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the PNG envelope is not pixels")


class TestTheFreeCameraFrame:
    def test_it_is_the_raw_rgb_buffer_at_the_size_asked_for(self) -> None:
        sim = _FakeSim()
        frame = _load().free_camera_frame(sim, 6, 4)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (4, 6, 3)
        assert frame.dtype == np.uint8
        # The free camera, at the CLI's --width/--height, not the camera's own size.
        assert sim.calls == [("default", 6, 4)]

    def test_the_frames_survive_the_first_last_spread_read(self) -> None:
        module = _load()
        first = module.free_camera_frame(_FakeSim(fill=10), 6, 4)
        last = module.free_camera_frame(_FakeSim(fill=13), 6, 4)
        # Exactly what _rollout reports after the loop; an envelope raised here.
        spread = float(np.mean(np.abs(last.astype(np.int16) - first.astype(np.int16))))
        assert spread == 3.0

    def test_the_rollout_never_collects_the_envelope(self) -> None:
        # Graded on the syntax tree, so the prose explaining the envelope is free
        # to name it while no call reaches it.
        tree = ast.parse(_EXAMPLE.read_text(encoding="utf-8"))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sim"
        }
        assert "get_frame" in called
        assert "render" not in called
