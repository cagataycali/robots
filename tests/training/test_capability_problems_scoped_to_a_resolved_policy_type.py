"""A capability problem describes a policy type that resolved, never one that did not.

:meth:`~strands_robots.training.lerobot.LerobotTrainer._validate_policy` refuses
a ``policy_type`` lerobot does not register, and then used to keep grading the
spec's capability knobs against it. Every capability probe answers off the
policy's own config class and falls back to a static set when the type is not in
the registry, and a misspelled name is in no such set - so each probe answered
"not supported" and the same response that said the policy does not exist went on
to describe its configuration, naming the types that DO expose the field (a list
containing the corrected spelling).

The remedy each of those problems carries is the harm: it prescribes dropping the
knob, and every misspelling below names a policy that supports it, so a caller
who followed the advice while fixing the name would have deleted a legitimate
setting.

The rule is the one :meth:`_validate_reward_model` already follows for
``extra['reward_model']``: a check that describes a config class is scoped to a
type that resolved, while checks that grade the REQUEST - a method spelling, a
tune key spelling, a flag's value, ``sample_weighting`` fields - stay
unconditional, being just as true for a misspelled policy.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.training.base import TrainSpec
from strands_robots.training.lerobot import (
    LerobotTrainer,
    _lerobot_policy_types,
    _policy_supports_embodiment_tag,
    _policy_supports_expert_only,
    _policy_supports_relative_actions,
    _policy_tune_components,
)

NOT_NATIVE = "is not LeRobot-native"
CAPABILITY = "supported by policy_type"

# (id, spec overrides, extra overrides, the policy the typo meant, the probe that
# proves the intended policy supports the knob the advice would have dropped).
TYPOS = [
    (
        "relative_actions",
        {},
        {"policy_type": "pi5", "relative_actions": True},
        "pi05",
        _policy_supports_relative_actions,
    ),
    ("embodiment", {"embodiment": "new_embodiment"}, {"policy_type": "grot"}, "groot", _policy_supports_embodiment_tag),
    ("tune_component", {"tune": {"llm": True}}, {"policy_type": "grot"}, "groot", _policy_tune_components),
    ("expert_only", {"method": "expert_only"}, {"policy_type": "smolvl"}, "smolvla", _policy_supports_expert_only),
]

# (id, extra overrides, spec overrides) - a request the spec gets wrong on its own
# terms, so the problem does not depend on the policy type resolving.
REQUEST_FAULTS = [
    ("tune_key_spelling", {}, {"tune": {"vision": True}}, "name no tunable component"),
    ("tune_value_domain", {}, {"tune": {"llm": "false"}}, "tune['llm']"),
    ("sample_weighting_field", {"sample_weighting": {"type": "rabc", "kapa": 0.01}}, {}, "does not support field(s)"),
    ("method_spelling", {}, {"method": "sideways"}, "unsupported method"),
]


@pytest.fixture
def spec(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return TrainSpec(
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        steps=10,
        global_batch_size=2,
        save_freq=5,
    )


def _problems(spec: TrainSpec, extra: dict, **overrides) -> list[str]:
    for field, value in overrides.items():
        setattr(spec, field, value)
    spec.extra = dict(spec.extra or {}) | extra
    return LerobotTrainer().validate(spec)


@pytest.mark.parametrize(("overrides", "extra"), [(o, e) for _, o, e, _, _ in TYPOS], ids=[i for i, *_ in TYPOS])
def test_a_misspelled_type_is_the_only_problem_reported(spec, overrides, extra):
    """The unresolved name is the one fault; its capabilities are not described."""
    problems = _problems(spec, extra, **overrides)
    assert len(problems) == 1, problems
    assert NOT_NATIVE in problems[0]
    assert CAPABILITY not in problems[0]


@pytest.mark.parametrize(("intended", "probe"), [(n, p) for _, _, _, n, p in TYPOS], ids=[i for i, *_ in TYPOS])
def test_the_intended_policy_supports_the_knob_the_old_advice_dropped(intended, probe):
    """Why suppressing the capability problem matters: the knob was legitimate.

    Each misspelling above is one edit away from a policy whose config declares
    the field, so "drop the knob or pick a supporting policy" pointed the caller
    away from the only real fault and at a setting they should keep.
    """
    assert intended in _lerobot_policy_types()
    assert probe(intended)


@pytest.mark.parametrize(
    ("overrides", "extra", "fragment"),
    [(o, e, f) for _, e, o, f in REQUEST_FAULTS],
    ids=[i for i, *_ in REQUEST_FAULTS],
)
def test_a_faulty_request_is_graded_whatever_the_type_is(spec, overrides, extra, fragment):
    """The type-independent checks are unaffected: both problems are reported."""
    problems = _problems(spec, {"policy_type": "grot"} | extra, **overrides)
    assert any(NOT_NATIVE in p for p in problems), problems
    assert any(fragment in p for p in problems), problems


@pytest.mark.parametrize(
    ("overrides", "extra", "fragment"),
    [
        ({}, {"relative_actions": True}, "relative_actions is not supported"),
        ({"embodiment": "new_embodiment"}, {}, "is not supported by policy_type 'act'"),
        ({"tune": {"llm": True}}, {}, "tune component(s) ['llm'] are not supported"),
        ({"method": "expert_only"}, {}, "method 'expert_only' is not supported"),
    ],
    ids=["relative_actions", "embodiment", "tune_component", "expert_only"],
)
def test_a_resolved_type_that_lacks_the_capability_is_still_refused(spec, overrides, extra, fragment):
    """The check is scoped, not removed: 'act' resolves and lacks all four."""
    problems = _problems(spec, {"policy_type": "act"} | extra, **overrides)
    assert not any(NOT_NATIVE in p for p in problems), problems
    assert any(fragment in p for p in problems), problems
