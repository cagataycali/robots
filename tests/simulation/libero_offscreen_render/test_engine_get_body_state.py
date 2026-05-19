"""Unit tests for ``LiberoOffScreenRenderEngine.get_body_state``.

Issue #170 added ``get_body_state`` to the engine so the BDDL goal
predicate evaluator can read body positions. Without it, the evaluator's
``_body_position`` returns ``None`` for every body and every
position-based predicate (``on``, ``inside``, ``near``, ``upright``)
silently returns ``False`` — manifested as the round-44 silent
counter bug for ``libero-10/SCENE5``.

These tests don't require running the real LIBERO env — they construct
the engine, inject a synthetic ``_env`` mock with the minimal interface
``get_body_state`` needs (``obj_body_id`` dict + ``sim.data.body_xpos`` /
``body_xquat`` arrays), and verify the returned status-dict format
matches what ``MuJoCoSimEngine.get_body_state`` returns. Heavy integration
coverage lives in ``tests_integ/benchmarks/libero/``.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="LiberoOffScreenRenderEngine module needs mujoco at construction")

from strands_robots.simulation.libero_offscreen_render.engine import (  # noqa: E402
    LiberoOffScreenRenderEngine,
)


class _FakeSimData:
    """Minimal robosuite-MjSim.data shape: array indexable by body_id."""

    def __init__(self, body_xpos: np.ndarray, body_xquat: np.ndarray) -> None:
        self.body_xpos = body_xpos
        self.body_xquat = body_xquat


class _FakeSimModel:
    """Minimal robosuite-MjSim.model shape: ``body_name2id(name) -> int``."""

    def __init__(self, name_to_id: dict[str, int]) -> None:
        self._name_to_id = name_to_id

    def body_name2id(self, name: str) -> int:
        return self._name_to_id.get(name, -1)


class _FakeSim:
    def __init__(self, data: _FakeSimData, model: _FakeSimModel) -> None:
        self.data = data
        self.model = model


class _FakeInnerEnv:
    """Stand-in for ``self._env.env`` (the underlying robosuite env).

    Holds the LIBERO-name → body-id mapping (``obj_body_id``) and the
    MjSim with body_xpos / body_xquat arrays.
    """

    def __init__(self, obj_body_id: dict[str, int], sim: _FakeSim) -> None:
        self.obj_body_id = obj_body_id
        self.sim = sim


class _FakeOffScreenEnv:
    """Stand-in for ``self._env`` (the OffScreenRenderEnv wrapper).

    Wraps an inner env (the actual robosuite env) under ``.env`` —
    matches LIBERO's ``ControlEnv`` / ``OffScreenRenderEnv`` two-layer
    structure.
    """

    def __init__(self, inner: _FakeInnerEnv) -> None:
        self.env = inner


def _make_engine(
    obj_body_id: dict[str, int],
    body_xpos: np.ndarray,
    body_xquat: np.ndarray,
    extra_mjcf_names: dict[str, int] | None = None,
) -> LiberoOffScreenRenderEngine:
    """Construct an engine with a fully-mocked underlying env.

    Bypasses ``setup_libero_task`` entirely.
    """
    name_to_id = {}
    # MJCF body name dictionary may include names NOT in obj_body_id
    # (engineered scene bodies the BDDL doesn't enumerate).
    if extra_mjcf_names:
        name_to_id.update(extra_mjcf_names)

    sim = _FakeSim(
        data=_FakeSimData(body_xpos=body_xpos, body_xquat=body_xquat),
        model=_FakeSimModel(name_to_id),
    )
    inner = _FakeInnerEnv(obj_body_id=obj_body_id, sim=sim)
    engine = LiberoOffScreenRenderEngine()
    engine._env = _FakeOffScreenEnv(inner)  # type: ignore[assignment]
    return engine


class TestGetBodyState:
    """#170: BDDL evaluator needs ``get_body_state`` on the engine to
    look up body positions for ``on`` / ``inside`` / ``near`` / etc.

    Without this method, every position-based predicate silently
    returns False even when the policy succeeds — caught by round-44
    instrumentation only because we side-by-sided against
    ``env.check_success``.
    """

    def test_resolves_via_obj_body_id(self):
        """LIBERO's BDDL-name → body-id dict is the primary lookup."""
        engine = _make_engine(
            obj_body_id={"porcelain_mug_1": 21, "plate_1": 24},
            body_xpos=np.zeros((30, 3)),
            body_xquat=np.zeros((30, 4)),
        )
        # Set known positions.
        engine._env.env.sim.data.body_xpos[21] = [-0.003, -0.311, 0.443]
        engine._env.env.sim.data.body_xquat[21] = [1.0, 0.0, 0.0, 0.0]

        result = engine.get_body_state("porcelain_mug_1")
        assert result["status"] == "success", result

        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        np.testing.assert_array_almost_equal(json_payload["position"], [-0.003, -0.311, 0.443])
        np.testing.assert_array_almost_equal(json_payload["quaternion"], [1.0, 0.0, 0.0, 0.0])

    def test_falls_back_to_body_name2id_when_not_in_obj_body_id(self):
        """For engineered scene bodies (e.g. ``robot0_base``) that aren't
        in LIBERO's ``obj_body_id`` dict, fall back to the raw MJCF
        ``body_name2id`` lookup."""
        engine = _make_engine(
            obj_body_id={"porcelain_mug_1": 21},  # only contains object bodies
            body_xpos=np.zeros((30, 3)),
            body_xquat=np.zeros((30, 4)),
            extra_mjcf_names={"robot0_base": 5},  # engineered body
        )
        engine._env.env.sim.data.body_xpos[5] = [0.0, 0.0, 0.0]
        engine._env.env.sim.data.body_xquat[5] = [1.0, 0.0, 0.0, 0.0]

        result = engine.get_body_state("robot0_base")
        assert result["status"] == "success", result

        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        np.testing.assert_array_almost_equal(json_payload["position"], [0.0, 0.0, 0.0])

    def test_unknown_body_returns_error(self):
        """Body absent from BOTH ``obj_body_id`` and the MJCF model
        produces a structured error rather than crashing or returning
        a stale position."""
        engine = _make_engine(
            obj_body_id={"porcelain_mug_1": 21},
            body_xpos=np.zeros((30, 3)),
            body_xquat=np.zeros((30, 4)),
        )
        result = engine.get_body_state("not_a_real_body")
        assert result["status"] == "error"
        assert "not_a_real_body" in result["content"][0]["text"]
        assert "not found" in result["content"][0]["text"].lower()

    def test_no_env_returns_error(self):
        """Calling before ``setup_libero_task`` returns an error rather
        than dereferencing None."""
        engine = LiberoOffScreenRenderEngine()
        result = engine.get_body_state("anything")
        assert result["status"] == "error"
        assert "not initialized" in result["content"][0]["text"].lower()

    def test_returned_format_matches_mujoco_engine_contract(self):
        """The returned dict format MUST match
        ``MuJoCoSimEngine.get_body_state`` so the predicate evaluator's
        ``_extract_json`` works unchanged. Specifically:

        - ``status`` key with ``"success"`` value
        - ``content`` list with at least one ``json`` block
        - ``json`` block contains ``position`` (3-list) and
          ``quaternion`` (4-list)
        """
        engine = _make_engine(
            obj_body_id={"cube_1": 0},
            body_xpos=np.array([[0.5, -0.1, 0.3]] + [[0, 0, 0]] * 29),
            body_xquat=np.array([[0.707, 0.707, 0.0, 0.0]] + [[1, 0, 0, 0]] * 29),
        )
        result = engine.get_body_state("cube_1")

        # Top-level keys.
        assert "status" in result
        assert "content" in result
        assert isinstance(result["content"], list)

        # JSON block schema.
        json_blocks = [c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c]
        assert len(json_blocks) >= 1
        payload = json_blocks[0]
        assert isinstance(payload.get("position"), list) and len(payload["position"]) == 3
        assert isinstance(payload.get("quaternion"), list) and len(payload["quaternion"]) == 4

    def test_works_with_predicate_evaluator(self):
        """End-to-end: a ``_body_on`` predicate compiled from BDDL
        should return True when the engine's ``get_body_state`` returns
        positions consistent with one body on top of another.

        Pin the integration so a future signature change to
        ``get_body_state`` (e.g., returning a different status-dict
        shape) gets caught by the predicate evaluator's expectations,
        not just the engine's unit test."""
        from strands_robots.benchmarks.libero.bddl_parser import compile_goal, parse_bddl

        engine = _make_engine(
            obj_body_id={"mug_1": 5, "plate_1": 6},
            body_xpos=np.zeros((10, 3)),
            body_xquat=np.zeros((10, 4)),
        )
        # mug 4mm above plate, 1cm xy off — the libero-10/SCENE5
        # at-success state.
        engine._env.env.sim.data.body_xpos[5] = [0.005, 0.0, 0.443]
        engine._env.env.sim.data.body_xpos[6] = [0.0, 0.0, 0.439]

        text = "(define (problem p) (:goal (on mug_1 plate_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        assert fn(engine) is True, (
            "BDDL evaluator should fire True when get_body_state returns "
            "positions consistent with mug-on-plate. If False, either the "
            "engine isn't returning the right format or _on_kwargs's tight "
            "tolerances regressed."
        )
