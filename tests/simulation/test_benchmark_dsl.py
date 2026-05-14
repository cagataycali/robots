"""Tests for ``strands_robots.simulation.benchmark_spec`` (declarative YAML/JSON loader).

Covers:

* :meth:`DeclarativeBenchmark.from_dict` schema validation (good / bad specs).
* :func:`register_benchmark_from_file` end-to-end with JSON + YAML.
* The sandboxed contract: unknown predicates / unknown top-level keys /
  non-dict predicate entries produce clear errors, not ``eval`` side-effects.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.benchmark import (
    _BENCHMARK_REGISTRY,
    get_benchmark,
)
from strands_robots.simulation.benchmark_spec import (
    DeclarativeBenchmark,
    register_benchmark_from_file,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


class _BodyStateSim:
    def __init__(self, positions: dict[str, list[float]]):
        self._pos = positions

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._pos:
            return {"status": "error", "content": [{"text": "missing"}]}
        return {
            "status": "success",
            "content": [
                {"text": body_name},
                {"json": {"position": self._pos[body_name]}},
            ],
        }

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


# Schema validation


class TestFromDictValidation:
    def test_minimal_valid_spec(self):
        spec = {
            "name": "minimal",
            "default_robot": "so100",
            "supported_robots": ["so100"],
        }
        bench = DeclarativeBenchmark.from_dict(spec)
        assert bench.name == "minimal"
        assert bench.default_robot == "so100"
        assert bench.supported_robots == ["so100"]
        assert bench.max_steps == 300  # default

    def test_rejects_non_dict_spec(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_unknown_top_level_keys(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(
                {"name": "x", "default_robot": "y", "supported_robots": ["y"], "weird_key": 1}
            )
        assert "weird_key" in str(exc.value)

    def test_rejects_missing_name(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict({"default_robot": "y", "supported_robots": []})

    def test_rejects_missing_default_robot(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict({"name": "x"})

    def test_rejects_default_not_in_supported(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "ghost", "supported_robots": ["a", "b"]})
        assert "not in supported_robots" in str(exc.value)

    def test_allows_default_outside_supported_when_empty(self):
        """Empty supported_robots means "any" - default outside makes sense."""
        bench = DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "anything", "supported_robots": []})
        assert bench.default_robot == "anything"

    def test_rejects_non_positive_max_steps(self):
        for bad in (-1, 0, "300", True):
            with pytest.raises(ValueError):
                DeclarativeBenchmark.from_dict(
                    {
                        "name": "x",
                        "default_robot": "y",
                        "supported_robots": ["y"],
                        "max_steps": bad,
                    }
                )


# Predicate compilation


class TestPredicateCompilation:
    def _base_spec(self, **overrides: Any) -> dict[str, Any]:
        spec = {
            "name": "t",
            "default_robot": "so100",
            "supported_robots": ["so100"],
            "max_steps": 10,
        }
        spec.update(overrides)
        return spec

    def test_success_all_true(self):
        spec = self._base_spec(
            success={
                "all": [
                    {"predicate": "body_above_z", "body": "cube", "z": 0.1},
                ]
            }
        )
        bench = DeclarativeBenchmark.from_dict(spec)
        sim_hit = _BodyStateSim({"cube": [0, 0, 0.2]})
        sim_miss = _BodyStateSim({"cube": [0, 0, 0.05]})
        assert bench.is_success(sim_hit) is True
        assert bench.is_success(sim_miss) is False

    def test_success_all_any_combined(self):
        """When both 'all' and 'any' are provided, both must hold."""
        spec = self._base_spec(
            success={
                "all": [{"predicate": "body_above_z", "body": "cube", "z": 0.0}],
                "any": [
                    {"predicate": "body_above_z", "body": "cube", "z": 10.0},
                    {"predicate": "body_above_z", "body": "cube", "z": 0.05},
                ],
            }
        )
        bench = DeclarativeBenchmark.from_dict(spec)
        sim = _BodyStateSim({"cube": [0, 0, 0.1]})
        # all: z>0.0 true. any: z>10 false OR z>0.05 true → any true. Combined: true.
        assert bench.is_success(sim) is True

    def test_failure_any(self):
        spec = self._base_spec(failure={"any": [{"predicate": "body_below_z", "body": "cube", "z": 0.0}]})
        bench = DeclarativeBenchmark.from_dict(spec)
        assert bench.is_failure(_BodyStateSim({"cube": [0, 0, -0.01]})) is True
        assert bench.is_failure(_BodyStateSim({"cube": [0, 0, 0.5]})) is False

    def test_dense_reward_sums_terms(self):
        spec = self._base_spec(
            dense_reward=[
                {"predicate": "constant", "value": 1.0},
                {"predicate": "constant", "value": -0.5},
            ]
        )
        bench = DeclarativeBenchmark.from_dict(spec)
        info = bench.on_step(None, {}, {})  # type: ignore[arg-type]
        assert info.reward == pytest.approx(0.5)

    def test_rejects_unknown_predicate(self):
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(success={"all": [{"predicate": "totally_made_up"}]}))
        assert "Unknown predicate" in str(exc.value)

    def test_rejects_non_dict_predicate_entry(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict(self._base_spec(success={"all": ["just a string"]}))

    def test_rejects_missing_predicate_key(self):
        with pytest.raises(ValueError):
            DeclarativeBenchmark.from_dict(self._base_spec(success={"all": [{"body": "cube", "z": 0.1}]}))

    def test_rejects_bad_clause_keys(self):
        """success/failure only allow 'all' / 'any'."""
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(self._base_spec(success={"all": [], "other": []}))
        assert "other" in str(exc.value)

    def test_predicate_bad_kwargs_surface_compile_error(self):
        """Bad predicate kwargs (wrong types, missing required) surface as a
        compile-time error, not a runtime predicate crash."""
        with pytest.raises(ValueError) as exc:
            DeclarativeBenchmark.from_dict(
                self._base_spec(success={"all": [{"predicate": "inside_region", "body": "x"}]})
            )
        # Should mention the predicate name in the error for discoverability.
        assert "inside_region" in str(exc.value)


# Empty / default clauses


class TestEmptyClauses:
    def test_success_absent_defaults_to_false(self):
        bench = DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "so100", "supported_robots": ["so100"]})
        assert bench.is_success(_BodyStateSim({})) is False

    def test_failure_absent_defaults_to_false(self):
        bench = DeclarativeBenchmark.from_dict({"name": "x", "default_robot": "so100", "supported_robots": ["so100"]})
        assert bench.is_failure(_BodyStateSim({})) is False

    def test_empty_success_returns_false(self):
        """Non-None but empty success clause must not default to "always true"."""
        bench = DeclarativeBenchmark.from_dict(
            {
                "name": "x",
                "default_robot": "so100",
                "supported_robots": ["so100"],
                "success": {"all": [], "any": []},
            }
        )
        assert bench.is_success(_BodyStateSim({})) is False


# File loading


class TestRegisterBenchmarkFromFile:
    def test_register_from_json(self, tmp_path):
        spec_path = tmp_path / "drawer.json"
        spec_path.write_text(
            json.dumps(
                {
                    "name": "drawer",
                    "default_robot": "so100",
                    "supported_robots": ["so100"],
                    "max_steps": 50,
                    "success": {
                        "all": [
                            {"predicate": "body_above_z", "body": "cube", "z": 0.1},
                        ]
                    },
                }
            )
        )
        bench = register_benchmark_from_file("drawer", str(spec_path))
        assert get_benchmark("drawer") is bench
        assert bench.max_steps == 50
        assert bench.is_success(_BodyStateSim({"cube": [0, 0, 0.5]})) is True

    def test_register_from_yaml(self, tmp_path):
        """YAML support is opt-in; skip if pyyaml isn't available in this env."""
        pytest.importorskip("yaml")
        spec_path = tmp_path / "y.yaml"
        spec_path.write_text(
            """
name: yml-task
default_robot: so100
supported_robots: [so100]
max_steps: 99
success:
  all:
    - {predicate: body_above_z, body: cube, z: 0.5}
"""
        )
        bench = register_benchmark_from_file("yml-task", str(spec_path))
        assert bench.max_steps == 99

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            register_benchmark_from_file("missing", str(tmp_path / "nope.json"))

    def test_rejects_unsupported_extension(self, tmp_path):
        p = tmp_path / "spec.toml"
        p.write_text("")
        with pytest.raises(ValueError) as exc:
            register_benchmark_from_file("x", str(p))
        assert ".toml" in str(exc.value) or "extension" in str(exc.value)

    def test_spec_name_internal_overridden_by_registry_name(self, tmp_path):
        """Registry name wins over any ``name`` declared inside the spec file."""
        p = tmp_path / "s.json"
        p.write_text(
            json.dumps(
                {
                    "name": "internal-name",
                    "default_robot": "so100",
                    "supported_robots": ["so100"],
                }
            )
        )
        register_benchmark_from_file("external-name", str(p))
        assert get_benchmark("external-name") is not None
        # The spec's internal name doesn't end up in the registry.
        assert get_benchmark("internal-name") is None

    def test_rejects_empty_name(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text('{"name": "x", "default_robot": "y", "supported_robots": []}')
        with pytest.raises(ValueError):
            register_benchmark_from_file("", str(p))

    def test_bad_json_propagates(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json}")
        with pytest.raises(json.JSONDecodeError):
            register_benchmark_from_file("x", str(p))


# DeclarativeBenchmark lifecycle


class TestDeclarativeBenchmarkLifecycle:
    def test_on_episode_start_delegates_to_base(self):
        """Default on_episode_start loads the default_robot when sim is empty."""
        spec = {
            "name": "x",
            "default_robot": "so100",
            "supported_robots": ["so100"],
        }
        bench = DeclarativeBenchmark.from_dict(spec)

        class FakeSim:
            def __init__(self):
                self.added: list[dict[str, Any]] = []

            def list_robots(self):
                return []

            def add_robot(self, *, name, data_config):
                self.added.append({"name": name, "data_config": data_config})

        sim = FakeSim()
        bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert len(sim.added) == 1
        assert sim.added[0]["data_config"] == "so100"

    def test_scene_load_error_raises(self, tmp_path: Path):
        """If the sim's load_scene returns an error dict, the benchmark must surface it."""
        spec = {
            "name": "x",
            "default_robot": "so100",
            "supported_robots": ["so100"],
            "scene": str(tmp_path / "missing.xml"),
        }
        bench = DeclarativeBenchmark.from_dict(spec)

        class FakeSim:
            def load_scene(self, path):
                return {"status": "error", "content": [{"text": f"no such file: {path}"}]}

            def list_robots(self):
                return ["preloaded"]

        sim = FakeSim()
        with pytest.raises(RuntimeError) as exc:
            bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert "load_scene" in str(exc.value)
