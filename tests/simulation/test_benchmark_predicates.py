"""Tests for ``strands_robots.simulation.predicates``.

Each predicate is tested against a lightweight fake sim that implements
only the methods the predicate exercises. Real MuJoCo integration is out
of scope here - those predicates are covered end-to-end in the dispatch
tests under ``tests/simulation/mujoco/``.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    make_predicate,
    register_predicate,
)

# Fake sim helpers


class _BodyStateSim:
    """Sim that exposes ``get_body_state`` with caller-provided positions."""

    def __init__(self, positions: dict[str, list[float]]):
        self._pos = positions

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._pos:
            return {"status": "error", "content": [{"text": f"Body '{body_name}' not found."}]}
        return {
            "status": "success",
            "content": [
                {"text": f"body {body_name}"},
                {
                    "json": {
                        "position": self._pos[body_name],
                        "quaternion": [1, 0, 0, 0],
                        "mass": 1.0,
                    }
                },
            ],
        }

    # Predicates that probe `get_observation` for joint state need this stub.
    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


class _JointObsSim:
    """Sim that exposes joint positions via ``get_observation``."""

    def __init__(self, joints: dict[str, float]):
        self._joints = joints

    def get_observation(self, *_, **__) -> dict[str, float]:
        return dict(self._joints)

    def get_body_state(self, body_name: str) -> dict[str, Any]:  # pragma: no cover
        return {"status": "error", "content": [{"text": "no bodies"}]}


class _ContactSim:
    """Sim that exposes ``get_contacts`` in the MuJoCo-backend shape."""

    def __init__(self, contacts: list[dict[str, Any]]):
        self._contacts = contacts

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {"text": f"{len(self._contacts)} contacts"},
                {
                    "json": {
                        "contacts": self._contacts,
                        "n_contacts": len(self._contacts),
                    }
                },
            ],
        }


class _NoHelpersSim:
    """Sim missing get_body_state / get_contacts entirely (e.g. future backend)."""

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


# Registry


class TestRegistry:
    def test_builtin_predicates_registered(self):
        required = {
            "body_above_z",
            "body_below_z",
            "joint_above",
            "joint_below",
            "distance_less_than",
            "inside_region",
            "contact_between",
            "contact_any",
            "body_on",
            "body_inside",
            "body_upright",
            "grasped",
            "distance_neg",
            "joint_progress",
            "constant",
        }
        assert required.issubset(PREDICATE_REGISTRY.keys())

    def test_make_predicate_unknown_raises(self):
        with pytest.raises(ValueError) as exc:
            make_predicate("totally_made_up")
        assert "Unknown predicate" in str(exc.value)
        # Error message should list valid names so the user can fix the spec.
        assert "body_above_z" in str(exc.value)

    def test_register_predicate_rejects_shadow(self):
        with pytest.raises(ValueError):
            register_predicate("body_above_z", lambda **_: lambda _sim: True)

    def test_register_predicate_rejects_non_callable(self):
        with pytest.raises(TypeError):
            register_predicate("my_pred", "not a callable")  # type: ignore[arg-type]

    def test_register_predicate_custom(self):
        try:

            def factory(value: float):
                return lambda _sim: value > 0

            register_predicate("positive_constant", factory)
            pred = make_predicate("positive_constant", value=1.5)
            assert pred(None) is True
        finally:
            PREDICATE_REGISTRY.pop("positive_constant", None)


# Body-position predicates


class TestBodyPositionPredicates:
    def test_body_above_z_true(self):
        sim = _BodyStateSim({"cube": [0.1, 0.0, 0.25]})
        pred = make_predicate("body_above_z", body="cube", z=0.2)
        assert pred(sim) is True

    def test_body_above_z_false(self):
        sim = _BodyStateSim({"cube": [0.1, 0.0, 0.15]})
        pred = make_predicate("body_above_z", body="cube", z=0.2)
        assert pred(sim) is False

    def test_body_above_z_missing_body_returns_false(self):
        sim = _BodyStateSim({"other": [0, 0, 1]})
        pred = make_predicate("body_above_z", body="cube", z=0.2)
        assert pred(sim) is False

    def test_body_below_z(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, -0.05]})
        pred = make_predicate("body_below_z", body="cube", z=0.0)
        assert pred(sim) is True

    def test_distance_less_than_true(self):
        sim = _BodyStateSim({"a": [0, 0, 0], "b": [0.05, 0, 0]})
        pred = make_predicate("distance_less_than", body_a="a", body_b="b", threshold=0.1)
        assert pred(sim) is True

    def test_distance_less_than_false(self):
        sim = _BodyStateSim({"a": [0, 0, 0], "b": [1.0, 0, 0]})
        pred = make_predicate("distance_less_than", body_a="a", body_b="b", threshold=0.1)
        assert pred(sim) is False

    def test_inside_region_matches(self):
        sim = _BodyStateSim({"cube": [0.1, 0.2, 0.3]})
        pred = make_predicate("inside_region", body="cube", min=[-0.5, 0.0, 0.0], max=[0.5, 0.5, 1.0])
        assert pred(sim) is True

    def test_inside_region_outside(self):
        sim = _BodyStateSim({"cube": [0.6, 0.0, 0.0]})
        pred = make_predicate("inside_region", body="cube", min=[0, 0, 0], max=[0.5, 0.5, 0.5])
        assert pred(sim) is False

    def test_inside_region_rejects_malformed_args(self):
        with pytest.raises(ValueError):
            make_predicate("inside_region", body="cube", min=[0, 0], max=[1, 1, 1])
        with pytest.raises(ValueError):
            # min > max should error up front, not silently always return False.
            make_predicate("inside_region", body="cube", min=[1, 1, 1], max=[0, 0, 0])

    def test_body_predicate_without_get_body_state_returns_false(self):
        sim = _NoHelpersSim()
        pred = make_predicate("body_above_z", body="cube", z=0)
        assert pred(sim) is False

    def test_body_position_libero_main_suffix_fallback(self):
        """Round 46 (#176 sub-task 3d) — LIBERO objects' BDDL names
        (``porcelain_mug_1``) map to MJCF root bodies suffixed with
        ``_main`` (``porcelain_mug_1_main``). The predicate evaluator
        must transparently retry with the suffix when the bare name
        misses, mirroring upstream LIBERO's
        ``env.objects_dict[name].root_body`` resolution. Without this,
        BDDL goal predicates like ``(On porcelain_mug_1 plate_1)``
        silently resolve to ``False`` even when physics has the mug
        on the plate.

        Pin: a sim that only exposes ``porcelain_mug_1_main`` (NOT
        ``porcelain_mug_1``) must still resolve via the predicate as
        if the bare name worked.
        """
        # Sim only knows the suffixed name (mimics MJCF body naming).
        sim = _BodyStateSim({"porcelain_mug_1_main": [0.0, 0.0, 0.5]})
        pred = make_predicate("body_above_z", body="porcelain_mug_1", z=0.4)
        assert pred(sim) is True, (
            "body_above_z with bare BDDL name should fall back to ``<name>_main`` "
            "for LIBERO scenes; round-46 fix may have regressed."
        )

    def test_body_position_main_suffix_no_double_suffix(self):
        """Already-suffixed names must not double-suffix on retry.
        Round 46 (#176 sub-task 3d).
        """
        # Sim only knows the suffixed name; caller passes already-suffixed.
        sim = _BodyStateSim({"plate_1_main": [0.0, 0.0, 0.4]})
        pred = make_predicate("body_above_z", body="plate_1_main", z=0.3)
        assert pred(sim) is True

    def test_body_position_bare_name_wins_over_suffix(self):
        """When BOTH ``<name>`` and ``<name>_main`` exist, prefer the
        bare lookup. This preserves the contract for fixtures /
        explicit-named bodies (e.g. ``living_room_table``) which don't
        use the LIBERO suffix.

        Round 46 (#176 sub-task 3d).
        """
        sim = _BodyStateSim(
            {
                "living_room_table": [0.0, 0.0, 0.46],
                "living_room_table_main": [99.0, 99.0, 99.0],  # decoy
            }
        )
        pred = make_predicate("body_above_z", body="living_room_table", z=0.4)
        assert pred(sim) is True
        # Decoy at 99.0 should not be reached if bare lookup wins.
        pred2 = make_predicate("body_above_z", body="living_room_table", z=98.0)
        assert pred2(sim) is False, "bare name should win over _main suffix; double-resolve detected"


# Joint predicates


class TestJointPredicates:
    def test_joint_above(self):
        sim = _JointObsSim({"drawer_slide": 0.18})
        assert make_predicate("joint_above", joint="drawer_slide", value=0.15)(sim) is True
        assert make_predicate("joint_above", joint="drawer_slide", value=0.2)(sim) is False

    def test_joint_below(self):
        sim = _JointObsSim({"gripper": 0.02})
        assert make_predicate("joint_below", joint="gripper", value=0.05)(sim) is True

    def test_joint_missing_returns_false(self):
        sim = _JointObsSim({"other_joint": 1.0})
        assert make_predicate("joint_above", joint="missing", value=0.0)(sim) is False

    def test_joint_progress_reward(self):
        sim = _JointObsSim({"drawer": 0.1})
        term = make_predicate("joint_progress", joint="drawer", target=0.2, weight=10.0)
        # -weight * |q - target| = -10 * 0.1 = -1.0
        assert term(sim) == pytest.approx(-1.0)

    def test_joint_progress_at_target_gives_zero_reward(self):
        sim = _JointObsSim({"drawer": 0.2})
        term = make_predicate("joint_progress", joint="drawer", target=0.2, weight=1.0)
        assert term(sim) == pytest.approx(0.0)


# Contact predicates


class TestContactPredicates:
    def test_contact_between_matches_either_order(self):
        sim = _ContactSim([{"geom1": "cube", "geom2": "gripper", "dist": -0.001}])
        assert make_predicate("contact_between", geom_a="cube", geom_b="gripper")(sim) is True
        assert make_predicate("contact_between", geom_a="gripper", geom_b="cube")(sim) is True

    def test_contact_between_no_match(self):
        sim = _ContactSim([{"geom1": "cube", "geom2": "ground"}])
        assert make_predicate("contact_between", geom_a="cube", geom_b="gripper")(sim) is False

    def test_contact_any(self):
        assert make_predicate("contact_any")(_ContactSim([{"geom1": "a", "geom2": "b"}])) is True
        assert make_predicate("contact_any")(_ContactSim([])) is False

    def test_contact_predicate_without_get_contacts(self):
        sim = _NoHelpersSim()
        assert make_predicate("contact_any")(sim) is False
        assert make_predicate("contact_between", geom_a="a", geom_b="b")(sim) is False


# Reward terms


class TestRewardTerms:
    def test_distance_neg_monotonic(self):
        far = _BodyStateSim({"a": [0, 0, 0], "b": [1, 0, 0]})
        near = _BodyStateSim({"a": [0, 0, 0], "b": [0.1, 0, 0]})
        term = make_predicate("distance_neg", body_a="a", body_b="b", weight=1.0)
        # Closer is greater (less negative).
        assert term(near) > term(far)

    def test_distance_neg_weight(self):
        sim = _BodyStateSim({"a": [0, 0, 0], "b": [1, 0, 0]})
        weighted = make_predicate("distance_neg", body_a="a", body_b="b", weight=5.0)
        assert weighted(sim) == pytest.approx(-5.0)

    def test_distance_neg_missing_body_returns_zero(self):
        """Missing bodies should not crash or reward heavily - return 0.0."""
        sim = _BodyStateSim({"a": [0, 0, 0]})
        term = make_predicate("distance_neg", body_a="a", body_b="ghost", weight=1.0)
        assert term(sim) == 0.0

    def test_constant(self):
        term = make_predicate("constant", value=-0.01)
        assert term(None) == pytest.approx(-0.01)


# LIBERO / #110 predicates


class _BodyStateWithQuatSim:
    """Extends _BodyStateSim with quaternion in the body-state payload."""

    def __init__(self, bodies: dict[str, dict[str, Any]]):
        self._bodies = bodies

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._bodies:
            return {"status": "error", "content": [{"text": "missing"}]}
        payload = {
            "position": self._bodies[body_name].get("position", [0, 0, 0]),
            "quaternion": self._bodies[body_name].get("quaternion", [1, 0, 0, 0]),
            "mass": 1.0,
        }
        return {"status": "success", "content": [{"text": body_name}, {"json": payload}]}

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


class TestBodyOn:
    def test_true_when_above_and_aligned(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 0.22], "table": [0.0, 0.0, 0.05]})
        pred = make_predicate("body_on", body_a="cube", body_b="table", z_offset=0.1)
        assert pred(sim) is True

    def test_false_when_not_above(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 0.04], "table": [0.0, 0.0, 0.05]})
        pred = make_predicate("body_on", body_a="cube", body_b="table", z_offset=0.01)
        assert pred(sim) is False

    def test_false_when_too_far_horizontally(self):
        sim = _BodyStateSim({"cube": [1.0, 0.0, 0.2], "table": [0.0, 0.0, 0.05]})
        pred = make_predicate("body_on", body_a="cube", body_b="table", xy_tol=0.1)
        assert pred(sim) is False

    def test_missing_body_returns_false(self):
        sim = _BodyStateSim({"table": [0, 0, 0.05]})
        pred = make_predicate("body_on", body_a="cube", body_b="table")
        assert pred(sim) is False


class TestBodyInside:
    def test_true_inside_box(self):
        sim = _BodyStateSim({"cube": [0.02, 0.01, 0.03], "basket": [0, 0, 0]})
        pred = make_predicate("body_inside", body="cube", container="basket", xy_tol=0.1, z_tol=0.1)
        assert pred(sim) is True

    def test_false_outside_xy(self):
        sim = _BodyStateSim({"cube": [0.5, 0.0, 0.0], "basket": [0, 0, 0]})
        pred = make_predicate("body_inside", body="cube", container="basket", xy_tol=0.1, z_tol=0.1)
        assert pred(sim) is False

    def test_false_outside_z(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 0.5], "basket": [0, 0, 0]})
        pred = make_predicate("body_inside", body="cube", container="basket", xy_tol=0.2, z_tol=0.1)
        assert pred(sim) is False


class TestBodyUpright:
    def test_identity_quat_is_upright(self):
        sim = _BodyStateWithQuatSim({"bottle": {"quaternion": [1.0, 0.0, 0.0, 0.0]}})
        pred = make_predicate("body_upright", body="bottle")
        assert pred(sim) is True

    def test_tipped_on_side_is_not_upright(self):
        # 90-deg rotation about x-axis: quat = (cos(pi/4), sin(pi/4), 0, 0) ≈ (0.707, 0.707, 0, 0)
        sim = _BodyStateWithQuatSim({"bottle": {"quaternion": [0.7071, 0.7071, 0.0, 0.0]}})
        pred = make_predicate("body_upright", body="bottle", tol=0.15)
        assert pred(sim) is False

    def test_small_tilt_within_tolerance(self):
        # Small rotation about x-axis - x component ~= 0.1, so 2*(x²+y²) ~= 0.02 < default tol 0.15.
        sim = _BodyStateWithQuatSim({"bottle": {"quaternion": [0.995, 0.1, 0.0, 0.0]}})
        pred = make_predicate("body_upright", body="bottle", tol=0.15)
        assert pred(sim) is True

    def test_missing_body_returns_false(self):
        sim = _BodyStateWithQuatSim({})
        pred = make_predicate("body_upright", body="bottle")
        assert pred(sim) is False

    def test_negative_tol_rejected(self):
        with pytest.raises(ValueError):
            make_predicate("body_upright", body="bottle", tol=-0.1)


class TestGrasped:
    def test_detects_gripper_contact_by_prefix(self):
        sim = _ContactSim(
            [
                {"geom1": "robot0_gripper_finger_r", "geom2": "cube_geom"},
            ]
        )
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is True

    def test_contact_without_gripper_prefix_is_not_grasp(self):
        sim = _ContactSim([{"geom1": "table", "geom2": "cube_geom"}])
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is False

    def test_matches_either_ordering(self):
        sim = _ContactSim(
            [
                {"geom1": "cube_geom", "geom2": "robot0_gripper_finger_l"},
            ]
        )
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is True

    def test_no_contacts_returns_false(self):
        sim = _ContactSim([])
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is False

    def test_without_get_contacts_returns_false(self):
        sim = _NoHelpersSim()
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is False
