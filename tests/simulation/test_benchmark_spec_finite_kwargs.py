"""Regression tests: a benchmark spec cannot smuggle a non-finite number in.

``benchmark_spec`` advertises itself as safe to load from "untrusted / LLM-authored"
files, and it is - for NAMES: the predicate registry is closed. But the remaining
keys were forwarded to the factory VERBATIM, and no factory checks them, so a
non-finite threshold or weight reached the reward. JSON spells it ``1e999``; YAML
spells it ``.inf``.

Measured end to end from an agent-authored file:

    dense_reward: [{predicate: constant, value: 1e999}]
    register_benchmark_from_file  -> success
    evaluate_benchmark            -> success, "Avg reward: inf"
    json                          -> avg_reward = inf

Also reachable as ``weight: 1e999`` on any float term (-> -inf). Every gate reported
success, so an eval run produced a meaningless score with nothing to flag it, and a
``nan`` reward poisons any training objective that consumes it.

Validation lives in ``_compile_call``, which every spec predicate flows through, so
``success`` / ``failure`` / ``dense_reward`` / ``stop_when`` are all covered.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark

_SUCCESS = {"all": [{"predicate": "body_above_z", "body": "c", "z": 0.5}]}


def _spec(**overrides):
    spec = {
        "name": "t",
        "instruction": "x",
        "default_robot": "panda",
        "supported_robots": ["panda"],
        "max_steps": 10,
        "success": dict(_SUCCESS),
        "dense_reward": [{"predicate": "constant", "value": 2.0}],
    }
    spec.update(overrides)
    return spec


def test_a_valid_spec_still_compiles() -> None:
    assert DeclarativeBenchmark.from_dict(_spec()) is not None


@pytest.mark.parametrize("literal", ["1e999", "-1e999"])
def test_an_infinite_reward_value_is_refused(literal) -> None:
    """The core defect: JSON has no inf literal, but 1e999 parses to one."""
    spec = json.loads(json.dumps(_spec()).replace('"value": 2.0', f'"value": {literal}'))
    with pytest.raises(ValueError, match="not finite"):
        DeclarativeBenchmark.from_dict(spec)


def test_an_infinite_weight_is_refused() -> None:
    """Any float term takes a weight, so this is the widest vector."""
    spec = _spec(
        dense_reward=[
            {
                "predicate": "distance_neg",
                "body_a": "c",
                "body_b": "panda/hand",
                "weight": float("inf"),
            }
        ]
    )
    with pytest.raises(ValueError, match="not finite"):
        DeclarativeBenchmark.from_dict(spec)


def test_a_nan_threshold_is_refused() -> None:
    spec = _spec(dense_reward=[{"predicate": "constant", "value": float("nan")}])
    with pytest.raises(ValueError, match="not finite"):
        DeclarativeBenchmark.from_dict(spec)


def test_a_success_clause_is_covered_too() -> None:
    """Validation sits in the shared compiler, not just the reward path."""
    spec = _spec(success={"all": [{"predicate": "body_above_z", "body": "c", "z": float("inf")}]})
    with pytest.raises(ValueError, match="not finite"):
        DeclarativeBenchmark.from_dict(spec)


def test_a_failure_clause_is_covered_too() -> None:
    spec = _spec(failure={"any": [{"predicate": "body_below_z", "body": "c", "z": float("-inf")}]})
    with pytest.raises(ValueError, match="not finite"):
        DeclarativeBenchmark.from_dict(spec)


def test_a_non_finite_element_inside_a_list_kwarg_is_refused() -> None:
    """Several predicates take sequences of floats, so check element-wise."""
    spec = _spec(success={"all": [{"predicate": "body_inside", "body": "c", "bounds": [0, 0, 0, 1, 1, float("inf")]}]})
    with pytest.raises(ValueError, match="not finite"):
        DeclarativeBenchmark.from_dict(spec)


def test_the_message_names_the_predicate_the_kwarg_and_the_index() -> None:
    """An agent has to be able to correct its own spec from the error alone."""
    spec = _spec(success={"all": [{"predicate": "body_inside", "body": "c", "bounds": [0, 0, 0, 1, 1, float("inf")]}]})
    with pytest.raises(ValueError) as excinfo:
        DeclarativeBenchmark.from_dict(spec)
    message = str(excinfo.value)
    assert "body_inside" in message
    assert "bounds[5]" in message


def test_a_bool_kwarg_is_not_mistaken_for_a_non_finite_number() -> None:
    """``bool`` is an ``int`` subclass; it is always finite and must pass through."""
    spec = _spec(success={"all": [{"predicate": "body_upright", "body": "c", "tol": 0.15}]})
    assert DeclarativeBenchmark.from_dict(spec) is not None


def test_a_string_kwarg_is_untouched(sim=None) -> None:
    """Entity names are strings and must not trip the numeric check."""
    spec = _spec(dense_reward=[{"predicate": "distance_neg", "body_a": "c", "body_b": "panda/hand"}])
    assert DeclarativeBenchmark.from_dict(spec) is not None


def test_a_large_but_finite_value_is_allowed() -> None:
    """The guard is finiteness, not magnitude - do not over-tighten."""
    spec = _spec(dense_reward=[{"predicate": "constant", "value": 1e30}])
    assert DeclarativeBenchmark.from_dict(spec) is not None


# ``staged_reward`` NESTS predicate calls and compiles them through
# ``make_predicate`` itself, bypassing the spec loader's own gate. Validating only
# in ``_compile_call`` therefore left the nested path open, which is why
# ``reject_non_finite_kwargs`` lives in ``predicates`` and both callers share it.

_OK_REWARD = {"predicate": "distance_neg", "body_a": "c", "body_b": "panda/hand"}
_OK_ADVANCE = {"predicate": "distance_less_than", "body_a": "c", "body_b": "panda/hand", "threshold": 0.05}


def _staged(stages):
    return _spec(dense_reward=[{"predicate": "staged_reward", "stages": stages}])


def test_a_valid_staged_reward_still_compiles() -> None:
    spec = _staged([{"reward": _OK_REWARD, "advance_when": _OK_ADVANCE}, {"reward": _OK_REWARD}])
    assert DeclarativeBenchmark.from_dict(spec) is not None


def test_a_nested_stage_reward_value_is_refused() -> None:
    """Measured: this registered and produced avg_reward = inf."""
    spec = _staged(
        [
            {"reward": {"predicate": "constant", "value": float("inf")}, "advance_when": _OK_ADVANCE},
            {"reward": _OK_REWARD},
        ]
    )
    with pytest.raises(ValueError, match="not finite"):
        DeclarativeBenchmark.from_dict(spec)


def test_a_nested_advance_when_threshold_is_refused() -> None:
    bad_advance = dict(_OK_ADVANCE, threshold=float("inf"))
    spec = _staged([{"reward": _OK_REWARD, "advance_when": bad_advance}, {"reward": _OK_REWARD}])
    with pytest.raises(ValueError, match="not finite"):
        DeclarativeBenchmark.from_dict(spec)


@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), float("nan")])
def test_a_non_finite_stage_bonus_is_refused(bad) -> None:
    """The bonus is ADDED the step a stage advances, so it lands in the reward.

    Measured with bonus=1e999: the staged term returned [inf, 0.0, 0.0].
    """
    spec = _staged([{"reward": _OK_REWARD, "advance_when": _OK_ADVANCE, "bonus": bad}, {"reward": _OK_REWARD}])
    with pytest.raises(ValueError, match="finite"):
        DeclarativeBenchmark.from_dict(spec)


def test_a_finite_stage_bonus_is_allowed() -> None:
    spec = _staged([{"reward": _OK_REWARD, "advance_when": _OK_ADVANCE, "bonus": 5.0}, {"reward": _OK_REWARD}])
    assert DeclarativeBenchmark.from_dict(spec) is not None


def test_the_nested_message_names_the_stage() -> None:
    """An agent must be able to find WHICH stage it got wrong."""
    spec = _staged(
        [
            {"reward": _OK_REWARD, "advance_when": _OK_ADVANCE},
            {"reward": {"predicate": "constant", "value": float("nan")}},
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        DeclarativeBenchmark.from_dict(spec)
    assert "stage[1]" in str(excinfo.value)


def test_a_staged_reward_built_directly_is_also_guarded() -> None:
    """The programmatic path must not be a bypass either."""
    from strands_robots.simulation.predicates import make_predicate

    stages = [
        {"reward": {"predicate": "constant", "value": float("inf")}, "advance_when": _OK_ADVANCE},
        {"reward": _OK_REWARD},
    ]
    with pytest.raises(ValueError, match="not finite"):
        make_predicate("staged_reward", stages=stages)
