"""The predicate/reward DSL rejects a reward term where a bool predicate is required.

A float-valued reward term used in a ``success`` / ``failure`` clause - or as a
``staged_reward`` ``advance_when`` gate - is read downstream as
``bool(<term(sim)>)``. A reward term returns a (usually nonzero) float, so
``bool(-0.5)`` is ``True``: the benchmark would silently report **instant
success** (or the phase machine would advance on step 0) with no error. These
tests pin that misuse to a clear load-time ``ValueError`` while leaving the
legitimate cases (bool predicate in ``success``, reward term OR sparse bool
predicate in ``dense_reward``) working.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.predicates import make_predicate, predicate_kind


def _spec(**overrides):
    spec = {"name": "kindguard", "default_robot": "so100", "supported_robots": ["so100"]}
    spec.update(overrides)
    return spec


class TestPredicateKind:
    """The classifier derives kind from the factory return annotation."""

    def test_bool_predicates_classify_as_bool(self):
        for n in ["body_above_z", "distance_less_than", "grasped", "body_upright", "contact_any"]:
            assert predicate_kind(n) == "bool", n

    def test_reward_terms_classify_as_float(self):
        for n in ["distance_neg", "joint_progress", "base_velocity", "base_orientation", "constant", "staged_reward"]:
            assert predicate_kind(n) == "float", n

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown predicate"):
            predicate_kind("totally_made_up")


class TestRewardTermRejectedInBoolClause:
    def test_reward_term_in_success_all_rejected(self):
        with pytest.raises(ValueError, match="reward term"):
            DeclarativeBenchmark.from_dict(
                _spec(success={"all": [{"predicate": "distance_neg", "body_a": "g", "body_b": "c"}]})
            )

    def test_reward_term_in_success_any_rejected(self):
        with pytest.raises(ValueError, match="reward term"):
            DeclarativeBenchmark.from_dict(_spec(success={"any": [{"predicate": "base_velocity", "vx": 0.5}]}))

    def test_reward_term_in_failure_rejected(self):
        with pytest.raises(ValueError, match="reward term"):
            DeclarativeBenchmark.from_dict(_spec(failure={"all": [{"predicate": "constant", "value": 1.0}]}))

    def test_bool_predicate_in_success_still_compiles(self):
        bench = DeclarativeBenchmark.from_dict(
            _spec(
                success={"all": [{"predicate": "distance_less_than", "body_a": "g", "body_b": "c", "threshold": 0.1}]}
            )
        )
        assert bench.name == "kindguard"

    def test_bool_predicate_in_dense_reward_allowed(self):
        # A bool predicate in dense_reward is a legitimate sparse 0/1 reward signal.
        bench = DeclarativeBenchmark.from_dict(
            _spec(dense_reward=[{"predicate": "grasped", "body": "c", "gripper_prefix": "robot0_gripper"}])
        )
        assert bench.name == "kindguard"

    def test_reward_term_in_dense_reward_allowed(self):
        bench = DeclarativeBenchmark.from_dict(
            _spec(dense_reward=[{"predicate": "distance_neg", "body_a": "g", "body_b": "c"}])
        )
        assert bench.name == "kindguard"

    def test_unknown_predicate_still_surfaces_verbatim(self):
        with pytest.raises(ValueError, match="Unknown predicate"):
            DeclarativeBenchmark.from_dict(_spec(success={"all": [{"predicate": "totally_made_up"}]}))


class TestStagedRewardAdvanceWhenKind:
    def _stages(self, advance_pred):
        return [
            {
                "reward": {"predicate": "distance_neg", "body_a": "g", "body_b": "c"},
                "advance_when": advance_pred,
            },
            {"reward": {"predicate": "constant", "value": 1.0}},
        ]

    def test_reward_term_advance_when_rejected(self):
        with pytest.raises(ValueError, match="reward term"):
            make_predicate(
                "staged_reward", stages=self._stages({"predicate": "distance_neg", "body_a": "g", "body_b": "c"})
            )

    def test_bool_advance_when_still_compiles(self):
        term = make_predicate(
            "staged_reward",
            stages=self._stages({"predicate": "distance_less_than", "body_a": "g", "body_b": "c", "threshold": 0.1}),
        )
        assert callable(term)
