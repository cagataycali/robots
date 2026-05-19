"""Unit tests for ``LiberoOffScreenRenderEngine.get_contacts``.

Issue #171 sub-task 3e replaced the engine's stub ``get_contacts``
(which returned a status='error' "not supported" message) with a real
implementation that reads from ``self._env.env.sim.data.contact[]``.
Required by the BDDL evaluator's contact-aware predicates (``_body_on``
with ``require_contact=True``, ``_grasped``, etc.) — without it, the
contact check silently no-ops and predicates fall back to
geometric-only verdicts (the pre-#171 behaviour, with transient false
positives during placement).

These tests don't require running the real LIBERO env — they construct
the engine, inject mock mujoco model + data with known contacts, and
verify the returned status-dict format matches what
``MuJoCoSimEngine.get_contacts`` returns.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="LiberoOffScreenRenderEngine module needs mujoco at construction")

from strands_robots.simulation.libero_offscreen_render.engine import (  # noqa: E402
    LiberoOffScreenRenderEngine,
)


class _FakeContact:
    """Mimics a single record from ``mujoco.MjData.contact[i]``."""

    def __init__(self, geom1: int, geom2: int, dist: float = 0.0, pos: list[float] | None = None):
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist
        self.pos = np.asarray(pos if pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)


class _FakeData:
    """Mimics ``mujoco.MjData`` for the slice ``get_contacts`` reads."""

    def __init__(self, contacts: list[_FakeContact]):
        self.ncon = len(contacts)
        self.contact = contacts


class _FakeModel:
    """Mimics ``mujoco.MjModel`` — an opaque token passed to mj_forward."""


class _FakeWrappedModel:
    """Robosuite's ``binding_utils.MjModel`` wraps the raw ``mujoco.MjModel`` under ``._model``."""

    def __init__(self, raw):
        self._model = raw


class _FakeWrappedData:
    """Robosuite's ``binding_utils.MjData`` wraps under ``._data``."""

    def __init__(self, raw):
        self._data = raw


class _FakeSim:
    def __init__(self, model_raw, data_raw):
        self.model = _FakeWrappedModel(model_raw)
        self.data = _FakeWrappedData(data_raw)


class _FakeInnerEnv:
    def __init__(self, sim):
        self.sim = sim


class _FakeOffScreenEnv:
    def __init__(self, inner):
        self.env = inner


def _make_engine(contacts: list[dict[str, object]]) -> LiberoOffScreenRenderEngine:
    """Build an engine with the given fake contacts wired through."""
    fake_contacts = [
        _FakeContact(
            geom1=int(c["geom1"]),  # type: ignore[call-overload]
            geom2=int(c["geom2"]),  # type: ignore[call-overload]
            dist=float(c.get("dist", 0.0)),  # type: ignore[arg-type]
            pos=list(c.get("pos", [0.0, 0.0, 0.0])),  # type: ignore[call-overload]
        )
        for c in contacts
    ]
    fake_model = _FakeModel()
    fake_data = _FakeData(fake_contacts)
    sim = _FakeSim(fake_model, fake_data)
    inner = _FakeInnerEnv(sim)
    engine = LiberoOffScreenRenderEngine()
    engine._env = _FakeOffScreenEnv(inner)  # type: ignore[assignment]
    return engine


@pytest.fixture
def patched_mujoco():
    """Patch ``mujoco.mj_forward`` (no-op on our fakes) and
    ``mujoco.mj_id2name`` (table-driven)."""
    geom_table: dict[int, str | None] = {}

    def fake_id2name(model, obj_type, obj_id):
        if obj_type == mujoco.mjtObj.mjOBJ_GEOM:
            return geom_table.get(obj_id)
        return None

    def fake_forward(model, data):
        # No-op — we don't need real physics.
        pass

    with (
        patch.object(mujoco, "mj_forward", side_effect=fake_forward),
        patch.object(mujoco, "mj_id2name", side_effect=fake_id2name),
    ):
        yield geom_table


class TestGetContacts:
    """#171 sub-task 3e: engine returns active contacts in
    MuJoCoSimEngine-compatible schema. The BDDL evaluator's
    ``_contact_between`` and ``_body_contact`` consume the same format.
    """

    def test_returns_empty_list_when_no_contacts(self, patched_mujoco):
        """No active contacts ⇒ empty list, status=success."""
        engine = _make_engine([])
        result = engine.get_contacts()
        assert result["status"] == "success"
        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        assert json_payload["contacts"] == []

    def test_returns_contact_with_geom_names(self, patched_mujoco):
        """Each contact record has ``geom1`` / ``geom2`` (string names),
        ``dist`` (float), ``pos`` (3-list). Schema matches
        ``MuJoCoSimEngine.get_contacts``."""
        patched_mujoco.update({1: "cube_1_g0", 2: "plate_1_g0"})
        engine = _make_engine([{"geom1": 1, "geom2": 2, "dist": -0.0001, "pos": [0.0, 0.0, 0.5]}])
        result = engine.get_contacts()
        assert result["status"] == "success"
        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        contacts = json_payload["contacts"]
        assert len(contacts) == 1
        c = contacts[0]
        assert c["geom1"] == "cube_1_g0"
        assert c["geom2"] == "plate_1_g0"
        assert isinstance(c["dist"], float)
        assert isinstance(c["pos"], list)
        assert len(c["pos"]) == 3

    def test_unnamed_geom_falls_back_to_id_string(self, patched_mujoco):
        """When a geom has no name (mj_id2name returns None), the record
        uses ``geom_<id>`` format. Defensive — robosuite-compiled MJCFs
        sometimes have unnamed visual geoms."""
        # No entries in geom_table ⇒ mj_id2name returns None.
        engine = _make_engine([{"geom1": 999, "geom2": 1000}])
        result = engine.get_contacts()
        assert result["status"] == "success"
        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        contacts = json_payload["contacts"]
        assert contacts[0]["geom1"] == "geom_999"
        assert contacts[0]["geom2"] == "geom_1000"

    def test_no_env_returns_error(self):
        """Calling before ``setup_libero_task`` returns structured error."""
        engine = LiberoOffScreenRenderEngine()
        result = engine.get_contacts()
        assert result["status"] == "error"
        assert "not initialized" in result["content"][0]["text"].lower()

    def test_returned_format_matches_mujoco_engine_contract(self, patched_mujoco):
        """Schema parity with ``MuJoCoSimEngine.get_contacts``: top-level
        ``status`` / ``content`` list with a ``json`` block whose
        ``contacts`` key is a list of contact dicts."""
        patched_mujoco.update({0: "a_g0", 1: "b_g0"})
        engine = _make_engine([{"geom1": 0, "geom2": 1}])
        result = engine.get_contacts()
        assert "status" in result
        assert isinstance(result["content"], list)
        assert any(isinstance(c, dict) and "json" in c for c in result["content"])
        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        assert "contacts" in json_payload
        assert isinstance(json_payload["contacts"], list)

    def test_used_by_predicate_evaluator(self, patched_mujoco):
        """End-to-end: a contact-aware ``on`` predicate compiled from
        BDDL fires True when the engine reports the matching contact.
        Pin the integration so a future schema change to
        ``get_contacts`` breaks both this test and the predicate
        evaluator at once (rather than only the predicate evaluator
        silently)."""
        from strands_robots.benchmarks.libero.bddl_parser import compile_goal, parse_bddl

        patched_mujoco.update({0: "cube_1_g0", 1: "plate_1_g0"})
        engine = _make_engine([{"geom1": 0, "geom2": 1}])

        # Also need ``get_body_state`` to work — patch it.
        def fake_get_body_state(self, body_name):
            positions = {
                "cube_1": [0.0, 0.0, 0.443],
                "plate_1": [0.0, 0.0, 0.439],
            }
            if body_name in positions:
                return {
                    "status": "success",
                    "content": [
                        {"text": ""},
                        {"json": {"position": positions[body_name], "quaternion": [1, 0, 0, 0]}},
                    ],
                }
            return {"status": "error", "content": [{"text": "missing"}]}

        with patch.object(LiberoOffScreenRenderEngine, "get_body_state", fake_get_body_state):
            text = "(define (problem p) (:goal (on cube_1 plate_1)))"
            problem = parse_bddl(text)
            fn = compile_goal(problem.goal)  # type: ignore[arg-type]
            assert fn(engine) is True, (
                "Contact-aware ``on`` predicate must fire True when "
                "engine.get_contacts reports a matching cube_1↔plate_1 contact"
            )
