"""The first minute on mjlab says what a new user would otherwise have to guess.

A fresh-eyes audit of the backend measured three silences on first contact:
``Robot("so101", backend="mjlab")`` with the default single world (47x slower
than the classic backend, nobody told), a cold Warp kernel cache (160 s of
JIT with no output), and 40-odd ``Module ... load on device`` lines from Warp
once the cache is warm. Each gets one line, or none, here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

pytest.importorskip("mjlab")
warp = pytest.importorskip("warp")

from strands_robots.simulation.mjlab import simulation as mjlab_sim  # noqa: E402
from strands_robots.simulation.mjlab.simulation import MjlabEngine  # noqa: E402

warp.init()  # resolves the real cache dir once, so the patched one below is read as-is


class TestSmallBatchIsSteered:
    def test_a_single_world_warns_and_names_the_classic_backend(self):
        with pytest.warns(UserWarning, match="backend='mujoco'") as rec:
            MjlabEngine(num_envs=1)
        assert "num_envs=1" in str(rec[0].message)
        assert "num_envs>=64" in str(rec[0].message)

    def test_below_the_threshold_warns_at_or_above_it_does_not(self):
        with pytest.warns(UserWarning):
            MjlabEngine(num_envs=mjlab_sim.SMALL_BATCH_WORLDS - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MjlabEngine(num_envs=mjlab_sim.SMALL_BATCH_WORLDS)
            MjlabEngine(num_envs=1024)


class TestColdKernelCacheIsAnnounced:
    def test_an_empty_cache_dir_is_cold(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(warp.config, "kernel_cache_dir", str(tmp_path))
        assert mjlab_sim.warp_kernel_cache_is_cold() == str(tmp_path)

    def test_a_cache_with_a_compiled_mujoco_warp_module_is_warm(self, tmp_path: Path, monkeypatch):
        (tmp_path / "wp_mujoco_warp._src.forward_0123abc").mkdir()
        monkeypatch.setattr(warp.config, "kernel_cache_dir", str(tmp_path))
        assert mjlab_sim.warp_kernel_cache_is_cold() is None

    def test_other_warp_modules_do_not_count_as_warm(self, tmp_path: Path, monkeypatch):
        (tmp_path / "wp___main___63adc70").mkdir()
        monkeypatch.setattr(warp.config, "kernel_cache_dir", str(tmp_path))
        assert mjlab_sim.warp_kernel_cache_is_cold() == str(tmp_path)

    def test_the_message_says_how_long_and_that_it_is_once(self):
        text = mjlab_sim.COLD_KERNEL_CACHE_MESSAGE.format(cache="/c")
        assert "first time" in text and "minutes" in text and "/c" in text


class TestWarpIsQuietByDefault:
    def test_the_info_default_becomes_warning(self, monkeypatch):
        monkeypatch.delenv("STRANDS_ROBOTS_LOG_LEVEL", raising=False)
        monkeypatch.setattr(warp.config, "log_level", warp.LOG_INFO)
        mjlab_sim.ensure_mjlab()
        assert warp.config.log_level == warp.LOG_WARNING

    def test_a_caller_s_own_level_is_kept(self, monkeypatch):
        monkeypatch.delenv("STRANDS_ROBOTS_LOG_LEVEL", raising=False)
        monkeypatch.setattr(warp.config, "log_level", warp.LOG_DEBUG)
        mjlab_sim.ensure_mjlab()
        assert warp.config.log_level == warp.LOG_DEBUG

    def test_debug_through_the_strands_env_keeps_warp_talking(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROBOTS_LOG_LEVEL", "debug")
        monkeypatch.setattr(warp.config, "log_level", warp.LOG_INFO)
        mjlab_sim.ensure_mjlab()
        assert warp.config.log_level == warp.LOG_INFO
