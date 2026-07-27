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


# Upstream-predicate registration + jitter default (issue: LIBERO dead e2e)


class TestUpstreamPredicateVocabulary:
    """Regression: real upstream LIBERO goals must register, not silently skip.

    Every ``libero_object`` task ships a goal of the form
    ``(:goal (And (In <obj> <container>)))`` (verified against the installed
    ``libero`` 0.1.1 corpus). The parser normalises the predicate head to
    lowercase, so the containment predicate must be registered under ``in``.
    Before the fix the vocabulary only carried ``inside``; ``(In ...)`` raised
    ``BDDLParseError`` and ``load_libero_suite`` returned ``{}`` (zero tasks).
    """

    def _libero_object_goal(self, obj: str, container: str) -> str:
        # Mirrors the pinned upstream BDDL goal casing/structure exactly.
        return (
            f"(define (problem LIBERO_Floor_Manipulation)\n"
            f"  (:language pick the {obj} and place it in the {container})\n"
            f"  (:objects {obj} - object {container} - object)\n"
            f"  (:goal (And (In {obj} {container}))))\n"
        )

    def test_libero_object_style_goal_registers(self, tmp_path):
        suite_dir = tmp_path / "libero_object"
        tasks = {
            "pick_up_the_alphabet_soup_and_place_it_in_the_basket": (
                "alphabet_soup_1",
                "basket_1_contain_region",
            ),
            "pick_up_the_cream_cheese_and_place_it_in_the_basket": (
                "cream_cheese_1",
                "basket_1_contain_region",
            ),
        }
        for stem, (obj, container) in tasks.items():
            _write(suite_dir / f"{stem}.bddl", self._libero_object_goal(obj, container))

        registered = load_libero_suite("libero_object", bddl_dir=suite_dir)

        # Full task set registers (pre-fix this was {} - zero tasks).
        assert len(registered) == len(tasks)
        assert set(registered) == {f"libero-object-{stem}" for stem in tasks}
        for name in registered:
            assert get_benchmark(name) is not None

    def test_in_goal_compiles_to_containment_predicate(self, tmp_path):
        suite_dir = tmp_path / "libero_object"
        _write(
            suite_dir / "t.bddl",
            self._libero_object_goal("milk_1", "basket_1_contain_region"),
        )
        registered = load_libero_suite("libero_object", bddl_dir=suite_dir)
        adapter = registered["libero-object-t"]
        # The compiled goal is a callable success predicate, not a no-op.
        assert callable(adapter._success_fn)


class TestSuiteJitterDefault:
    """Regression: suite default jitter must match the adapter's own default.

    A non-zero suite default silently perturbed every task's canonical init
    state, putting checkpoints such as ``nvidia/GR00T-N1.7-LIBERO`` out of
    distribution even for callers who never asked for jitter.
    """

    def test_suite_default_matches_adapter_default(self):
        import inspect

        from strands_robots.benchmarks.libero.adapter import LiberoAdapter

        suite_default = inspect.signature(load_libero_suite).parameters["init_jitter"].default
        adapter_default = inspect.signature(LiberoAdapter.__init__).parameters["init_jitter"].default
        assert suite_default == adapter_default == 0.0

    def test_registered_adapters_have_zero_jitter_by_default(self, tmp_path):
        suite_dir = tmp_path / "libero_spatial"
        _write(suite_dir / "t.bddl", "(define (problem t) (:goal (grasped cube)))")
        registered = load_libero_suite("libero_spatial", bddl_dir=suite_dir)
        assert registered["libero-spatial-t"]._init_jitter == 0.0


# Package-root resolution (_resolve_libero_root)


class _FakeLibero:
    """Minimal stand-in for the installed ``libero`` package object.

    ``require_optional`` imports ``libero`` and inspects ``__file__`` /
    ``__path__`` to locate the BDDL tree, so a stub carrying only those
    attributes is enough to drive every branch of ``_resolve_libero_root``
    without the real package on disk.
    """

    def __init__(self, file=None, path=None):
        if file is not None:
            self.__file__ = file
        if path is not None:
            self.__path__ = path


@pytest.fixture
def _patch_require_optional(monkeypatch):
    """Return a setter that makes ``_resolve_libero_root`` see a chosen stub.

    ``_resolve_libero_root`` pulls the package via ``require_optional``; patching
    that single seam keeps the test off the real import system and its cache.
    """

    def _set(stub):
        monkeypatch.setattr(
            "strands_robots.benchmarks.libero.suite.require_optional",
            lambda *a, **k: stub,
        )

    return _set


class TestResolveLiberoRoot:
    def test_regular_package_uses_file_grandparent(self, _patch_require_optional):
        """A normal install (``__file__`` set) roots at the dir above the package."""
        from strands_robots.benchmarks.libero.suite import _resolve_libero_root

        _patch_require_optional(_FakeLibero(file="/opt/site-packages/libero/__init__.py"))
        from pathlib import Path

        assert _resolve_libero_root() == Path("/opt/site-packages")

    def test_namespace_package_uses_path_entry_parent(self, _patch_require_optional):
        """A PEP 420 namespace install (``__file__`` None) roots off ``__path__``."""
        from strands_robots.benchmarks.libero.suite import _resolve_libero_root

        _patch_require_optional(_FakeLibero(file=None, path=["/checkout/libero"]))
        from pathlib import Path

        assert _resolve_libero_root() == Path("/checkout")

    def test_no_file_and_empty_path_raises(self, _patch_require_optional):
        """Neither ``__file__`` nor a usable ``__path__`` is an actionable error."""
        from strands_robots.benchmarks.libero.suite import _resolve_libero_root

        _patch_require_optional(_FakeLibero(file=None, path=[]))
        with pytest.raises(RuntimeError, match="neither __file__ nor"):
            _resolve_libero_root()


# BDDL-dir probing fallback (_locate_bddl_dir without override)


class TestLocateBddlDirProbe:
    def test_probes_standard_layout_when_no_override(self, tmp_path, monkeypatch):
        """With no ``bddl_dir=`` the loader walks the package layout and finds the suite."""
        # Lay out one of the candidate paths: <root>/libero/bddl_files/<suite>.
        suite_dir = tmp_path / "libero" / "bddl_files" / "libero_spatial"
        _write(suite_dir / "pick_cube.bddl", "(define (problem t) (:goal (grasped cube)))")
        monkeypatch.setattr(
            "strands_robots.benchmarks.libero.suite._resolve_libero_root",
            lambda: tmp_path,
        )
        registered = load_libero_suite("libero_spatial", load_init_states=False)
        assert "libero-spatial-pick_cube" in registered

    def test_raises_when_no_candidate_layout_matches(self, tmp_path, monkeypatch):
        """If none of the probed layouts exist, the error lists what was tried."""
        from strands_robots.benchmarks.libero.suite import _locate_bddl_dir

        monkeypatch.setattr(
            "strands_robots.benchmarks.libero.suite._resolve_libero_root",
            lambda: tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="Could not locate BDDL directory"):
            _locate_bddl_dir("libero_spatial", None)


# Init-state loading (_load_init_states_by_bddl) error and happy paths


class _FakeTaskSuite:
    """Stand-in for an instantiated LIBERO benchmark suite object."""

    def __init__(self, bddl_files, init_states, *, bddl_raises=False, num_tasks=None):
        self._bddl_files = bddl_files
        self._init_states = init_states
        self._bddl_raises = bddl_raises
        self._num_tasks = num_tasks if num_tasks is not None else len(bddl_files)

    def get_num_tasks(self):
        return self._num_tasks

    def get_task_bddl_files(self):
        if self._bddl_raises:
            raise RuntimeError("bddl listing exploded")
        return self._bddl_files

    def get_task_init_states(self, task_id):
        states = self._init_states[task_id]
        if isinstance(states, Exception):
            raise states
        return states


def _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict):
    """Register fake ``libero.libero.benchmark`` modules in ``sys.modules``.

    ``_load_init_states_by_bddl`` does ``from libero.libero import benchmark``;
    seeding the three package levels lets the test drive its branches without
    the real LIBERO package. ``get_benchmark_dict`` may be a callable or
    ``None`` to exercise the missing-attribute path.
    """
    import sys
    import types

    pkg = types.ModuleType("libero")
    pkg.__path__ = []  # mark as package
    sub = types.ModuleType("libero.libero")
    sub.__path__ = []
    bench = types.ModuleType("libero.libero.benchmark")
    if get_benchmark_dict is not None:
        bench.get_benchmark_dict = get_benchmark_dict
    monkeypatch.setitem(sys.modules, "libero", pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", sub)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", bench)


class TestLoadInitStatesByBddl:
    def test_libero_not_importable_returns_empty(self, monkeypatch):
        """When ``libero`` cannot be imported the map is empty (adapter falls back)."""
        import sys

        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        # Ensure the import fails even if libero is installed in the env.
        monkeypatch.setitem(sys.modules, "libero.libero", None)
        assert _load_init_states_by_bddl("libero_spatial") == {}

    def test_missing_get_benchmark_dict_returns_empty(self, monkeypatch):
        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict=None)
        assert _load_init_states_by_bddl("libero_spatial") == {}

    def test_get_benchmark_dict_raising_returns_empty(self, monkeypatch):
        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        def _boom():
            raise RuntimeError("registry init failed")

        _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict=_boom)
        assert _load_init_states_by_bddl("libero_spatial") == {}

    def test_unknown_suite_returns_empty(self, monkeypatch):
        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict=lambda: {})
        assert _load_init_states_by_bddl("libero_spatial") == {}

    def test_suite_factory_raising_returns_empty(self, monkeypatch):
        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        def _factory():
            raise RuntimeError("cannot build suite")

        _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict=lambda: {"libero_spatial": _factory})
        assert _load_init_states_by_bddl("libero_spatial") == {}

    def test_get_task_bddl_files_raising_returns_empty(self, monkeypatch):
        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        ts = _FakeTaskSuite(bddl_files=["a.bddl"], init_states=[object()], bddl_raises=True)
        _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict=lambda: {"libero_spatial": lambda: ts})
        assert _load_init_states_by_bddl("libero_spatial") == {}

    def test_happy_path_keys_on_bare_bddl_filename(self, monkeypatch):
        """Each task's init states are keyed by the bare BDDL filename."""
        import numpy as np

        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        states0 = np.zeros((50, 79), dtype=np.float32)
        states1 = np.ones((50, 84), dtype=np.float32)
        ts = _FakeTaskSuite(
            bddl_files=["/pkg/bddl/pick_cube.bddl", "/pkg/bddl/stack_block.bddl"],
            init_states=[states0, states1],
        )
        _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict=lambda: {"libero_spatial": lambda: ts})
        out = _load_init_states_by_bddl("libero_spatial")
        assert set(out) == {"pick_cube.bddl", "stack_block.bddl"}
        assert out["pick_cube.bddl"].shape == (50, 79)
        assert out["stack_block.bddl"].shape == (50, 84)

    def test_per_task_init_state_failure_skips_only_that_task(self, monkeypatch):
        """One task raising in ``get_task_init_states`` must not drop the others."""
        import numpy as np

        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        good = np.zeros((50, 79), dtype=np.float32)
        ts = _FakeTaskSuite(
            bddl_files=["/pkg/good.bddl", "/pkg/bad.bddl"],
            init_states=[good, RuntimeError("missing init-state file")],
        )
        _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict=lambda: {"libero_spatial": lambda: ts})
        out = _load_init_states_by_bddl("libero_spatial")
        assert set(out) == {"good.bddl"}

    def test_task_id_beyond_bddl_files_is_skipped(self, monkeypatch):
        """A ``num_tasks`` larger than the bddl-file list skips the overflow ids."""
        import numpy as np

        from strands_robots.benchmarks.libero.suite import _load_init_states_by_bddl

        good = np.zeros((50, 79), dtype=np.float32)
        ts = _FakeTaskSuite(
            bddl_files=["/pkg/only_one.bddl"],
            init_states=[good],
            num_tasks=3,
        )
        _install_fake_libero_benchmark(monkeypatch, get_benchmark_dict=lambda: {"libero_spatial": lambda: ts})
        out = _load_init_states_by_bddl("libero_spatial")
        assert set(out) == {"only_one.bddl"}
