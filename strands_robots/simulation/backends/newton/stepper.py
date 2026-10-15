"""``num_envs`` copies of one compiled MuJoCo model on the GPU, stepped by MuJoCo-Warp (one CUDA graph per step)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...._extras import hint

HINT = f"physics='newton' needs mujoco_warp + warp: {hint('newton')}"


class NewtonStepper:
    def __init__(self, device: str | None, num_envs: int) -> None:
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as exc:
            raise ImportError(f"{HINT} ({exc})") from exc
        self.mjw, self.wp = mjw, wp
        wp.init()
        self.device = device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
        self.num_envs = num_envs
        self.m: Any = None
        self.d: Any = None
        self._model: Any = None
        self._graph: Any = None

    def load(self, model: Any, data: Any) -> None:
        """Put the model on the device and broadcast ``data`` to every env."""
        with self.wp.ScopedDevice(self.device):
            self._model = model
            self.m = self.mjw.put_model(model)
            self.d = self.mjw.put_data(model, data, nworld=self.num_envs)
            self._capture()

    def _capture(self) -> None:
        """Record one step into a CUDA graph (kernel-by-kernel launches are ~10x slower); CPU devices run eagerly."""
        self._graph = None
        if not str(self.device).startswith("cuda"):
            return
        try:
            self.mjw.step(self.m, self.d)  # compiles every kernel before the capture
            with self.wp.ScopedCapture() as cap:
                self.mjw.step(self.m, self.d)
            self._graph = cap.graph
        except Exception:  # noqa: BLE001 — the graph is an optimisation; eager stepping is the contract
            self._graph = None

    def step(self, n: int) -> None:
        with self.wp.ScopedDevice(self.device):
            for _ in range(n):
                if self._graph is not None:
                    self.wp.capture_launch(self._graph)
                else:
                    self.mjw.step(self.m, self.d)

    def set_ctrl(self, ids: list[int], targets: np.ndarray) -> None:
        """The same targets in every env, written in place (the captured graph keeps the buffer)."""
        ctrl = self.d.ctrl.numpy()
        ctrl[:, ids] = targets
        self.d.ctrl.assign(ctrl)

    def sync_to(self, data: Any) -> None:
        """Env 0 into the CPU ``MjData`` (positions, velocities, time), so reads and renders see the GPU state."""
        import mujoco

        data.qpos[:] = self.d.qpos.numpy()[0]
        data.qvel[:] = self.d.qvel.numpy()[0]
        data.time = float(self.d.time.numpy()[0])
        mujoco.mj_forward(self._model, data)

    def sync_from(self, data: Any) -> None:
        """The CPU ``MjData`` into every env (after a reset, a teleport or a scene change)."""
        self.load(self._model, data)

    def close(self) -> None:
        self.m = self.d = self._graph = None
