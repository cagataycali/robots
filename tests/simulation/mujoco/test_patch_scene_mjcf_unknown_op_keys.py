"""``patch_scene_mjcf`` refuses op keys it does not read.

Every field of every structured op is read with a fallback default (``pos``
defaults to the origin, ``quat`` to identity, ``type`` to ``"box"``, ``parent``
to ``"world"``), so a key outside an op's vocabulary is not an inert extra: the
op runs with that default and reports success. A misspelled ``pos`` moved a body
to the world origin, a misspelled ``parent`` re-parented it to the worldbody,
and a misspelled ``type`` compiled a box where a sphere was requested - each
reported ``status="success"`` with an "op(s) applied" count.

These tests pin the corrected contract: an unrecognised key is refused, the
message names the op, the key and the keys that op accepts, and - because the
batch is atomic - the scene is left exactly as it was.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    _PATCH_OP_KEYS,
    _unknown_op_keys_error,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_patch_keys", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


CRATE_POS = [0.4, 0.0, 0.5]


def _seeded_world(sim: Simulation) -> None:
    """Compile a world holding one named body ``crate`` at :data:`CRATE_POS`."""
    sim.create_world()
    result = sim.patch_scene_mjcf(
        [
            {"op": "add_body", "parent": "world", "name": "crate", "pos": CRATE_POS},
            {"op": "add_geom", "body": "crate", "type": "box", "size": [0.2, 0.2, 0.2]},
        ]
    )
    assert result["status"] == "success"


def _text(result: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _body_pos(sim: Simulation, name: str) -> list[float]:
    mj = sim._mj
    assert sim._world is not None and sim._world._model is not None
    body_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"body {name!r} missing"
    return [float(v) for v in sim._world._model.body_pos[body_id]]


def _body_names(sim: Simulation) -> list[str]:
    mj = sim._mj
    assert sim._world is not None and sim._world._model is not None
    model = sim._world._model
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]


class TestAMisspelledFieldDoesNotApplyItsDefault:
    """The silent-default cases: each of these used to report success."""

    def test_set_body_pos_with_position_key_does_not_move_the_body(self, sim: Simulation) -> None:
        _seeded_world(sim)

        result = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "position": [0.4, 0.0, 0.9]}])

        assert result["status"] == "error"
        assert "'position'" in _text(result)
        # Pre-fix this reported "1 op(s) applied" and left the crate at the origin.
        assert _body_pos(sim, "crate") == CRATE_POS

    def test_set_body_quat_with_quaternion_key_does_not_reset_orientation(self, sim: Simulation) -> None:
        sim.create_world()
        assert (
            sim.patch_scene_mjcf(
                [{"op": "add_body", "name": "crate", "pos": CRATE_POS, "quat": [0.7071, 0.0, 0.7071, 0.0]}]
            )["status"]
            == "success"
        )
        mj = sim._mj
        assert sim._world is not None and sim._world._model is not None
        body_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "crate")
        before = [float(v) for v in sim._world._model.body_quat[body_id]]

        result = sim.patch_scene_mjcf([{"op": "set_body_quat", "name": "crate", "quaternion": [1.0, 0.0, 0.0, 0.0]}])

        assert result["status"] == "error"
        assert "'quaternion'" in _text(result)
        after = [float(v) for v in sim._world._model.body_quat[body_id]]
        assert after == pytest.approx(before)

    def test_add_geom_with_shape_key_does_not_compile_the_default_box(self, sim: Simulation) -> None:
        _seeded_world(sim)
        assert sim._world is not None and sim._world._model is not None
        geoms_before = sim._world._model.ngeom

        result = sim.patch_scene_mjcf([{"op": "add_geom", "body": "crate", "shape": "sphere", "size": [0.2, 0.2, 0.2]}])

        assert result["status"] == "error"
        assert "'shape'" in _text(result)
        assert sim._world._model.ngeom == geoms_before

    def test_add_body_with_misspelled_pos_does_not_spawn_at_the_origin(self, sim: Simulation) -> None:
        _seeded_world(sim)

        result = sim.patch_scene_mjcf([{"op": "add_body", "name": "widget", "postion": [1.0, 1.0, 1.0]}])

        assert result["status"] == "error"
        assert "'postion'" in _text(result)
        assert "widget" not in _body_names(sim)

    def test_add_body_with_misspelled_parent_does_not_reparent_to_world(self, sim: Simulation) -> None:
        _seeded_world(sim)

        result = sim.patch_scene_mjcf([{"op": "add_body", "name": "child", "parent_body": "crate"}])

        assert result["status"] == "error"
        assert "'parent_body'" in _text(result)
        assert "child" not in _body_names(sim)

    def test_add_site_with_misspelled_pos_does_not_place_it_at_the_body_origin(self, sim: Simulation) -> None:
        _seeded_world(sim)
        assert sim._world is not None and sim._world._model is not None
        sites_before = sim._world._model.nsite

        result = sim.patch_scene_mjcf([{"op": "add_site", "body": "crate", "name": "tip", "location": [0, 0, 0.3]}])

        assert result["status"] == "error"
        assert "'location'" in _text(result)
        assert sim._world._model.nsite == sites_before

    def test_delete_body_keyed_by_body_does_not_silently_target_nothing(self, sim: Simulation) -> None:
        _seeded_world(sim)

        result = sim.patch_scene_mjcf([{"op": "delete_body", "body": "crate"}])

        assert result["status"] == "error"
        assert "crate" in _body_names(sim)


class TestTheRejectionIsActionable:
    def test_message_names_the_op_the_key_and_the_accepted_keys(self, sim: Simulation) -> None:
        _seeded_world(sim)

        text = _text(sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "position": [0, 0, 1]}]))

        assert "set_body_pos" in text
        assert "'position' (did you mean 'pos'?)" in text
        assert "Accepted keys: name, op, pos." in text

    def test_an_unrelated_key_gets_the_accepted_list_without_a_bogus_suggestion(self, sim: Simulation) -> None:
        _seeded_world(sim)

        # "group" is a real MJCF site attribute this op vocabulary does not
        # cover: it must be refused, but nothing here is a plausible typo of it.
        text = _text(sim.patch_scene_mjcf([{"op": "add_site", "body": "crate", "name": "tip", "group": 2}]))

        assert "'group'" in text
        assert "did you mean" not in text
        assert "Accepted keys: body, name, op, pos, rgba, size." in text

    def test_every_unknown_key_in_one_op_is_reported_together(self, sim: Simulation) -> None:
        _seeded_world(sim)

        text = _text(
            sim.patch_scene_mjcf(
                [{"op": "add_geom", "body": "crate", "shape": "sphere", "color": [0, 0, 1, 1], "size": [0.1] * 3}]
            )
        )

        assert "'color'" in text and "'shape'" in text

    def test_a_bad_key_late_in_a_batch_rolls_the_whole_batch_back(self, sim: Simulation) -> None:
        _seeded_world(sim)

        result = sim.patch_scene_mjcf(
            [
                {"op": "set_body_pos", "name": "crate", "pos": [0.1, 0.1, 0.1]},
                {"op": "add_body", "name": "widget", "postion": [1, 1, 1]},
            ]
        )

        assert result["status"] == "error"
        assert "patch op #2 failed" in _text(result)
        assert _body_pos(sim, "crate") == CRATE_POS
        assert "widget" not in _body_names(sim)


class TestTheDocumentedVocabularyIsAccepted:
    def test_an_op_using_every_key_it_documents_is_applied(self, sim: Simulation) -> None:
        sim.create_world()

        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "parent": "world", "name": "crate", "pos": CRATE_POS, "quat": [1, 0, 0, 0]},
                {
                    "op": "add_geom",
                    "body": "crate",
                    "type": "box",
                    "size": [0.2, 0.2, 0.2],
                    "rgba": [0.9, 0.2, 0.1, 1.0],
                    "name": "crate_geom",
                    "pos": [0, 0, 0],
                    "quat": [1, 0, 0, 0],
                },
                {
                    "op": "add_site",
                    "body": "crate",
                    "name": "tip",
                    "pos": [0, 0, 0.2],
                    "size": [0.01],
                    "rgba": [0, 1, 0, 1],
                },
                {"op": "set_body_quat", "name": "crate", "quat": [1, 0, 0, 0]},
                {"op": "set_body_pos", "name": "crate", "pos": CRATE_POS},
                {"op": "delete_body", "name": "crate"},
            ]
        )

        assert result["status"] == "success"
        assert "6 op(s) applied" in _text(result)

    @pytest.mark.parametrize("kind", sorted(_PATCH_OP_KEYS))
    def test_the_op_key_itself_is_part_of_every_vocabulary(self, kind: str) -> None:
        # Guards against an entry that forgets "op" and so rejects every call.
        assert "op" in _PATCH_OP_KEYS[kind]
        assert _unknown_op_keys_error(kind, {"op": kind}) is None


class TestTheKeyCheckIsIndependentOfValueValidation:
    def test_a_non_string_key_is_refused(self) -> None:
        # A JSON-shaped op reaching the tool can carry a non-string key; it must
        # be reported like any other unknown key rather than crashing the lookup.
        op: dict[Any, Any] = {"op": "set_body_pos", 3: [0, 0, 1]}

        message = _unknown_op_keys_error("set_body_pos", op)

        assert message is not None
        assert "3" in message

    def test_a_fully_documented_op_passes_the_key_check(self) -> None:
        assert _unknown_op_keys_error("add_geom", {"op": "add_geom", "body": "b", "type": "sphere"}) is None
