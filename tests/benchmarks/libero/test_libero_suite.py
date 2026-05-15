"""Tests for :func:`load_libero_suite` and the suite enumeration helpers.

These tests do NOT require the ``libero`` pip package - they all use the
``bddl_dir=`` override to point at a temp directory of hand-written BDDL
files. The upstream-package path is covered indirectly (via the probe
fallback in :func:`_locate_bddl_dir`) but not exercised directly; that
requires the real package layout and would bloat CI.
"""

from __future__ import annotations

import pytest

from strands_robots.benchmarks.libero.suite import (
    SUITE_NAMES,
    _normalise_suite_name,
    available_suites,
    load_libero_suite,
)
from strands_robots.simulation.benchmark import _BENCHMARK_REGISTRY, get_benchmark


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# Suite name normalisation


class TestSuiteNames:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("libero_spatial", "libero_spatial"),
            ("libero-spatial", "libero_spatial"),
            ("spatial", "libero_spatial"),
            ("LIBERO-10", "libero_10"),
            ("  libero_90  ", "libero_90"),
        ],
    )
    def test_normalise(self, raw, expected):
        assert _normalise_suite_name(raw) == expected

    def test_available_suites_matches_SUITE_NAMES(self):
        assert set(available_suites()) == set(SUITE_NAMES)


# load_libero_suite with bddl_dir override


class TestLoadLiberoSuite:
    def test_registers_all_tasks_under_prefix(self, tmp_path):
        suite_dir = tmp_path / "libero_spatial"
        _write(
            suite_dir / "pick_up_the_red_cube.bddl",
            "(define (problem t1) (:goal (on cube plate)))",
        )
        _write(
            suite_dir / "stack_blue_block.bddl",
            "(define (problem t2) (:goal (on block base)))",
        )

        registered = load_libero_suite("libero_spatial", bddl_dir=suite_dir)
        assert set(registered.keys()) == {
            "libero-spatial-pick_up_the_red_cube",
            "libero-spatial-stack_blue_block",
        }
        # Each one is retrievable from the global registry.
        assert get_benchmark("libero-spatial-pick_up_the_red_cube") is not None

    def test_custom_key_prefix(self, tmp_path):
        suite_dir = tmp_path / "libero_object"
        _write(suite_dir / "task_a.bddl", "(define (problem t) (:goal (grasped a)))")
        registered = load_libero_suite("libero_object", bddl_dir=suite_dir, key_prefix="")
        assert "object-task_a" in registered

    def test_resolves_scene_path_when_file_exists(self, tmp_path):
        suite_dir = tmp_path / "libero_spatial"
        scene_dir = tmp_path / "scenes"
        _write(suite_dir / "pick_cube.bddl", "(define (problem t) (:goal (grasped cube)))")
        _write(scene_dir / "pick_cube.xml", "<mujoco/>")

        registered = load_libero_suite("libero_spatial", bddl_dir=suite_dir, scene_dir=scene_dir)
        adapter = registered["libero-spatial-pick_cube"]
        assert adapter.scene_path == str(scene_dir / "pick_cube.xml")

    def test_missing_scene_leaves_adapter_scene_none(self, tmp_path):
        suite_dir = tmp_path / "libero_spatial"
        scene_dir = tmp_path / "scenes"
        scene_dir.mkdir()
        _write(suite_dir / "pick_cube.bddl", "(define (problem t) (:goal (grasped cube)))")

        registered = load_libero_suite("libero_spatial", bddl_dir=suite_dir, scene_dir=scene_dir)
        adapter = registered["libero-spatial-pick_cube"]
        assert adapter.scene_path is None

    def test_malformed_bddl_is_skipped_not_fatal(self, tmp_path, caplog):
        """A single bad BDDL file must not prevent the rest of the suite from loading."""
        suite_dir = tmp_path / "libero_spatial"
        _write(suite_dir / "good.bddl", "(define (problem good) (:goal (grasped cube)))")
        _write(suite_dir / "bad.bddl", "(this is not bddl")

        with caplog.at_level("WARNING"):
            registered = load_libero_suite("libero_spatial", bddl_dir=suite_dir)
        assert "libero-spatial-good" in registered
        assert "libero-spatial-bad" not in registered
        assert any("Skipping" in rec.message for rec in caplog.records)

    def test_forwards_max_steps_and_jitter(self, tmp_path):
        suite_dir = tmp_path / "libero_spatial"
        _write(suite_dir / "t.bddl", "(define (problem t) (:goal (grasped cube)))")

        registered = load_libero_suite("libero_spatial", bddl_dir=suite_dir, max_steps=42, init_jitter=0.0)
        adapter = registered["libero-spatial-t"]
        assert adapter.max_steps == 42
        assert adapter._init_jitter == 0.0

    def test_unknown_suite_name_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="libero_"):
            load_libero_suite("libero_unknown_suite", bddl_dir=tmp_path)

    def test_nonexistent_bddl_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_libero_suite("libero_spatial", bddl_dir=tmp_path / "nope")

    def test_empty_directory_registers_nothing(self, tmp_path):
        suite_dir = tmp_path / "libero_spatial"
        suite_dir.mkdir()
        registered = load_libero_suite("libero_spatial", bddl_dir=suite_dir)
        assert registered == {}
