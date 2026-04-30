"""Tests for :class:`VideoConfig` and the run_policy video-param consolidation.

Previously, ``SimEngine.run_policy`` had 5 flat video parameters
(``record_video``, ``video_fps``, ``video_camera``, ``video_width``,
``video_height``). They are now folded into a single typed
:class:`VideoConfig` on ``PolicyRunner.run`` and a ``video: dict``
kwarg on ``SimEngine.run_policy``.

This file locks:

1. ``VideoConfig`` dataclass contract (defaults, ``enabled``, ``from_dict``).
2. ``VideoConfig.from_dict`` accepts both canonical and legacy keys.
3. ``SimEngine.run_policy`` signature no longer exposes flat video params.
4. The MuJoCo dispatcher folds legacy tool_spec keys
   (``output_path``/``fps``/``camera_name``) into ``video`` automatically.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots.simulation.policy_runner import VideoConfig


class TestVideoConfigDataclass:
    def test_default_config_is_disabled(self) -> None:
        cfg = VideoConfig()
        assert cfg.path is None
        assert cfg.enabled is False
        assert cfg.fps == 30
        assert cfg.camera is None
        assert cfg.width == 640
        assert cfg.height == 480

    def test_enabled_when_path_set(self) -> None:
        assert VideoConfig(path="/tmp/x.mp4").enabled is True

    def test_enabled_false_for_empty_string(self) -> None:
        """Empty path must be treated as "no recording", not a valid path."""
        assert VideoConfig(path="").enabled is False

    def test_frozen(self) -> None:
        cfg = VideoConfig(path="/tmp/a.mp4")
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.fps = 60  # type: ignore[misc]


class TestVideoConfigFromDict:
    def test_none_passthrough(self) -> None:
        assert VideoConfig.from_dict(None) is None

    def test_empty_dict_passthrough(self) -> None:
        assert VideoConfig.from_dict({}) is None

    def test_canonical_keys(self) -> None:
        cfg = VideoConfig.from_dict({"path": "/tmp/a.mp4", "fps": 60, "camera": "wrist", "width": 320, "height": 240})
        assert cfg is not None
        assert cfg.path == "/tmp/a.mp4"
        assert cfg.fps == 60
        assert cfg.camera == "wrist"
        assert cfg.width == 320
        assert cfg.height == 240

    def test_legacy_record_video_alias(self) -> None:
        """Back-compat: the old ``record_video`` flat kwarg name is accepted."""
        cfg = VideoConfig.from_dict({"record_video": "/tmp/legacy.mp4"})
        assert cfg is not None
        assert cfg.path == "/tmp/legacy.mp4"

    def test_legacy_output_path_alias(self) -> None:
        """tool_spec.json uses ``output_path``; legacy callers accepted."""
        cfg = VideoConfig.from_dict({"output_path": "/tmp/spec.mp4", "fps": 24})
        assert cfg is not None
        assert cfg.path == "/tmp/spec.mp4"
        assert cfg.fps == 24

    def test_legacy_video_fps_alias(self) -> None:
        cfg = VideoConfig.from_dict({"path": "/tmp/a.mp4", "video_fps": 15})
        assert cfg is not None
        assert cfg.fps == 15


class TestRunPolicySignatureNoFlatVideoParams:
    """Regression: the ABC and MuJoCo override must not expose flat video params."""

    _FORBIDDEN = {"record_video", "video_fps", "video_camera", "video_width", "video_height"}

    def test_sim_engine_run_policy_has_only_video_param(self) -> None:
        from strands_robots.simulation.base import SimEngine

        params = inspect.signature(SimEngine.run_policy).parameters
        leaked = self._FORBIDDEN.intersection(params)
        assert not leaked, f"SimEngine.run_policy still exposes flat video params: {leaked}"
        assert "video" in params

    def test_mujoco_run_policy_has_only_video_param(self) -> None:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        params = inspect.signature(Simulation.run_policy).parameters
        leaked = self._FORBIDDEN.intersection(params)
        assert not leaked, f"MuJoCo run_policy still exposes flat video params: {leaked}"
        assert "video" in params

    def test_policy_runner_run_has_only_video_param(self) -> None:
        from strands_robots.simulation.policy_runner import PolicyRunner

        params = inspect.signature(PolicyRunner.run).parameters
        leaked = self._FORBIDDEN.intersection(params)
        assert not leaked, f"PolicyRunner.run still exposes flat video params: {leaked}"
        assert "video" in params


class TestDispatcherFoldsFlatVideoKeys:
    """Agent callers pass flat ``output_path``/``fps`` via tool_spec.json.

    The MuJoCo dispatcher must fold those into a ``video`` dict before
    calling ``run_policy``, so Python-level and agent-level callers end
    up on the same code path.

    We subclass ``Simulation`` and override ``run_policy`` with the exact
    same signature so ``inspect.signature`` in the dispatcher matches
    against the real parameter list.
    """

    def _make_capturing_sim(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        captured: dict = {}

        class _CapturingSim(Simulation):
            def run_policy(  # type: ignore[override]
                self,
                robot_name: str,
                policy_provider: str = "mock",
                policy_config: dict | None = None,
                instruction: str = "",
                duration: float = 10.0,
                control_frequency: float = 50.0,
                action_horizon: int = 8,
                fast_mode: bool = False,
                video: dict | None = None,
            ) -> dict:
                captured.update(
                    {
                        "robot_name": robot_name,
                        "policy_provider": policy_provider,
                        "policy_config": policy_config,
                        "instruction": instruction,
                        "duration": duration,
                        "control_frequency": control_frequency,
                        "action_horizon": action_horizon,
                        "fast_mode": fast_mode,
                        "video": video,
                    }
                )
                return {"status": "success", "content": [{"text": "ok"}]}

        sim = _CapturingSim.__new__(_CapturingSim)
        return sim, captured

    def test_dispatcher_folds_flat_keys(self) -> None:
        sim, captured = self._make_capturing_sim()
        sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm0",
                "output_path": "/tmp/x.mp4",
                "fps": 25,
                "camera_name": "wrist",
            },
        )
        assert captured["video"] == {"path": "/tmp/x.mp4", "fps": 25, "camera": "wrist"}

    def test_dispatcher_no_path_no_video(self) -> None:
        """Without ``output_path``, dispatcher must pass ``video=None``."""
        sim, captured = self._make_capturing_sim()
        sim._dispatch_action(
            "run_policy",
            {"robot_name": "arm0", "fps": 25, "camera_name": "wrist"},
        )
        assert captured["video"] is None, "dispatcher must not synthesise a video dict without an output path"

    def test_dispatcher_passes_explicit_video_dict_through(self) -> None:
        """If caller already provides ``video`` explicitly, don't clobber it."""
        sim, captured = self._make_capturing_sim()
        explicit_video = {"path": "/tmp/explicit.mp4", "fps": 120}
        sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm0",
                "video": explicit_video,
                "output_path": "/tmp/should_be_ignored.mp4",  # explicit wins
            },
        )
        assert captured["video"] == explicit_video
