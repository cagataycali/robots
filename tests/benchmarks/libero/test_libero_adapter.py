"""Tests for :class:`LiberoAdapter`.

Covers:

* Construction via ``from_file`` / ``from_text`` / raw ``__init__``.
* ``supported_robots`` / ``default_robot`` = Panda-only.
* ``instruction`` surfaces the BDDL ``:language`` string.
* ``is_success`` positive + negative cases against fake sims (no MuJoCo
  needed - the predicates poll ``get_body_state`` / ``get_contacts``).
* ``on_episode_start`` loads the scene (or errors cleanly) and applies
  per-episode jitter when the sim exposes ``move_object``.
* Integration with ``PolicyRunner.evaluate`` and
  ``SimEngine.evaluate_benchmark`` via a minimal ``FakeSim`` stub.
* Error surface: unknown task via ``evaluate_benchmark`` returns a
  structured error dict, never raises.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from strands_robots.benchmarks.libero import (
    BDDLParseError,
    LiberoAdapter,
)
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import (
    _BENCHMARK_REGISTRY,
    register_benchmark,
)
from strands_robots.simulation.policy_runner import PolicyRunner


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


# Representative BDDL fragments

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""

COMPOUND_BDDL = """
(define (problem libero_grasp_and_upright)
  (:language "grasp the bottle and keep it upright")
  (:goal (and (grasped bottle_1) (upright bottle_1))))
"""

NEGATED_BDDL = """
(define (problem libero_release)
  (:goal (not (grasped cube_1))))
"""


# Fake sim helpers


class _FakeRobot:
    def __init__(self, data_config: str):
        self.data_config = data_config


class _FakeWorld:
    def __init__(self, robots: dict[str, _FakeRobot]):
        self.robots = dict(robots)


class FakeSim(SimEngine):
    """Minimal ``SimEngine`` with get_body_state / get_contacts / move_object."""

    def __init__(
        self,
        bodies: dict[str, dict[str, Any]] | None = None,
        contacts: list[dict[str, str]] | None = None,
        data_config: str = "panda",
    ):
        self._bodies = dict(bodies or {})
        self._contacts = list(contacts or [])
        self._reset_count = 0
        self._move_calls: list[tuple[str, list[float]]] = []
        self._world = _FakeWorld({"fake_panda": _FakeRobot(data_config)})
        self._scenes_loaded: list[str] = []

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self._reset_count += 1
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {}

    def add_robot(self, name, **kw):
        dc = kw.get("data_config") or "panda"
        self._world.robots[name] = _FakeRobot(dc)
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self):
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name):
        return ["j0", "j1"]

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self.robot_joint_names(robot_name or "fake_panda")}

    def send_action(self, action, robot_name=None, n_substeps=1):
        pass

    def render(self, camera_name="default", width=None, height=None):
        return {"status": "success", "content": [{"text": "render"}]}

    # Optional helpers used by predicates + adapter

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._bodies:
            return {"status": "error", "content": [{"text": "missing"}]}
        return {
            "status": "success",
            "content": [
                {"text": body_name},
                {
                    "json": {
                        "position": self._bodies[body_name].get("position", [0, 0, 0]),
                        "quaternion": self._bodies[body_name].get("quaternion", [1, 0, 0, 0]),
                        "mass": 1.0,
                    }
                },
            ],
        }

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {"text": f"{len(self._contacts)} contacts"},
                {"json": {"contacts": self._contacts, "n_contacts": len(self._contacts)}},
            ],
        }

    def move_object(self, *, name: str, position: list[float]) -> dict[str, Any]:
        self._move_calls.append((name, list(position)))
        self._bodies.setdefault(name, {})["position"] = list(position)
        return {"status": "success"}

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        self._scenes_loaded.append(scene_path)
        return {"status": "success"}


# Construction


class TestConstruction:
    def test_from_text_happy_path(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        assert adapter.supported_robots == ["panda"]
        assert adapter.default_robot == "panda"
        assert adapter.max_steps == 300
        assert adapter.instruction == "pick up the red cube and place it on the plate"

    def test_from_text_respects_max_steps_override(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, max_steps=75)
        assert adapter.max_steps == 75

    def test_from_text_rejects_negative_jitter(self):
        with pytest.raises(ValueError):
            LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=-0.1)

    def test_rejects_bddl_without_goal(self):
        text = '(define (problem no_goal) (:language "no goal block"))'
        with pytest.raises(ValueError, match="no \\(:goal"):
            LiberoAdapter.from_text(text)

    def test_from_file(self, tmp_path):
        p = tmp_path / "task.bddl"
        p.write_text(PICK_CUBE_BDDL)
        adapter = LiberoAdapter.from_file(p)
        assert adapter.problem.name == "libero_spatial_pick_cube"

    def test_from_text_propagates_parse_errors(self):
        with pytest.raises(BDDLParseError):
            LiberoAdapter.from_text("(define (problem p) (:goal (telekinesis cube_1)))")


# Lifecycle hooks


class TestIsSuccess:
    def test_positive_case_on(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.25]},
                "plate_1": {"position": [0, 0, 0.1]},
            }
        )
        assert adapter.is_success(sim) is True

    def test_negative_case_on(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.05]},  # not above plate
                "plate_1": {"position": [0, 0, 0.1]},
            }
        )
        assert adapter.is_success(sim) is False

    def test_compound_goal_needs_both(self):
        adapter = LiberoAdapter.from_text(COMPOUND_BDDL)
        sim_neither = FakeSim(bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}})
        sim_upright_only = FakeSim(bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}})
        sim_both = FakeSim(
            bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}},
            contacts=[{"geom1": "robot0_gripper_finger_l", "geom2": "bottle_1"}],
        )
        assert adapter.is_success(sim_neither) is False
        assert adapter.is_success(sim_upright_only) is False
        assert adapter.is_success(sim_both) is True

    def test_negated_goal(self):
        adapter = LiberoAdapter.from_text(NEGATED_BDDL)
        sim_empty = FakeSim()
        sim_gripped = FakeSim(contacts=[{"geom1": "robot0_gripper_finger_l", "geom2": "cube_1"}])
        assert adapter.is_success(sim_empty) is True
        assert adapter.is_success(sim_gripped) is False


class TestOnEpisodeStart:
    def test_loads_scene_before_compat_check(self, tmp_path):
        """``scene_path`` load must happen before ``super().on_episode_start``
        so the base compat check sees the scene's Panda robot."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, scene_path="/fake/scene.xml")
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))
        assert sim._scenes_loaded == ["/fake/scene.xml"]

    def test_scene_load_error_raises(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, scene_path="/bad/path.xml")

        sim = FakeSim(data_config="panda")
        # Override load_scene to report failure.
        sim.load_scene = lambda path: {  # type: ignore[assignment]
            "status": "error",
            "content": [{"text": f"no such file: {path}"}],
        }
        with pytest.raises(RuntimeError, match="load_scene"):
            adapter.on_episode_start(sim, random.Random(0))

    def test_jitter_applied_when_init_declares_subject(self):
        adapter = LiberoAdapter.from_text(
            """
            (define (problem p)
              (:init (on cube_1 table_1))
              (:goal (on cube_1 plate_1)))
            """,
            init_jitter=0.01,
        )
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0.0, 0.0, 0.1]},
                "table_1": {"position": [0.0, 0.0, 0.0]},
            },
            data_config="panda",
        )
        adapter.on_episode_start(sim, random.Random(42))
        # cube_1 got jittered; table_1 is only the reference and not moved.
        assert any(call[0] == "cube_1" for call in sim._move_calls)
        assert not any(call[0] == "table_1" for call in sim._move_calls)

    def test_jitter_disabled_when_zero(self):
        adapter = LiberoAdapter.from_text(
            """
            (define (problem p)
              (:init (on cube_1 table_1))
              (:goal (on cube_1 plate_1)))
            """,
            init_jitter=0.0,
        )
        sim = FakeSim(
            bodies={"cube_1": {"position": [0, 0, 0.1]}, "table_1": {"position": [0, 0, 0]}},
            data_config="panda",
        )
        adapter.on_episode_start(sim, random.Random(42))
        assert sim._move_calls == []

    def test_non_panda_robot_rejected_by_base_compat_check(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(data_config="so100")  # not panda
        from strands_robots.simulation.benchmark import BenchmarkCompatibilityError

        with pytest.raises(BenchmarkCompatibilityError) as exc:
            adapter.on_episode_start(sim, random.Random(0))
        assert exc.value.supported == ["panda"]
        assert exc.value.data_config == "so100"


# Step semantics


class TestOnStep:
    def test_sparse_reward_zero_and_not_done(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        info = adapter.on_step(FakeSim(), {}, {})
        assert info.reward == 0.0
        assert info.done is False


# PolicyRunner + evaluate_benchmark integration


class TestEvaluateBenchmarkIntegration:
    def test_evaluate_with_mock_policy_succeeds(self):
        """Mock policy drives a loop; the benchmark loop returns a success_rate
        without crashing even though the mock policy doesn't actually win."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, max_steps=4)
        register_benchmark("libero-test-pick", adapter)
        # Sim is loaded with Panda but predicate positions don't match the
        # goal, so every episode should fall through to max_steps with
        # success=False - that's fine, we're testing the loop.
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.0]},
                "plate_1": {"position": [0, 0, 0.0]},
            },
            data_config="panda",
        )
        # Rename so SimEngine.evaluate_benchmark can resolve the sole robot.
        sim._world.robots.clear()
        sim._world.robots["panda_arm"] = _FakeRobot("panda")

        result = sim.evaluate_benchmark(
            benchmark_name="libero-test-pick",
            policy_provider="mock",
            n_episodes=2,
            seed=0,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_episodes"] == 2
        assert payload["benchmark_class"] == "LiberoAdapter"

    def test_unknown_task_returns_structured_error(self):
        sim = FakeSim(data_config="panda")
        result = sim.evaluate_benchmark(benchmark_name="libero-nonexistent")
        assert result["status"] == "error"
        assert "no benchmark registered" in result["content"][0]["text"].lower()

    def test_runner_counts_success_when_predicate_holds(self):
        """Seed the sim with predicate-satisfying state so is_success returns
        True on the first step - success_rate should be 1.0."""
        adapter = LiberoAdapter.from_text(
            "(define (problem p) (:goal (upright bottle)))",
            max_steps=3,
        )
        sim = FakeSim(
            bodies={"bottle": {"quaternion": [1.0, 0, 0, 0]}},
            data_config="panda",
        )
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_panda"))
        result = PolicyRunner(sim).evaluate("fake_panda", policy, spec=adapter, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_success"] == 1
        assert payload["episodes"][0]["steps"] == 1  # terminated on first step
