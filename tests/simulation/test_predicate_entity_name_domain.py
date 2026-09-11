"""A predicate-DSL kwarg that names a scene entity is refused when it cannot name one.

The sibling of ``test_predicate_kwarg_finiteness`` and
``test_predicate_tolerance_sign_domain``: those hold the NUMERIC kwargs of a
clause to a domain because a value outside it leaves the clause permanently
decided. The kwargs that name a scene entity - a body, a joint, a geom, a
container, and ``grasped``'s ``gripper_prefix`` - carried no domain at all, and
``_kwarg_domain_error``'s own docstring said so ("a ``str`` body name ... is
untouched").

Measured on the pre-fix tree, SO-101 in MuJoCo, a 25 mm cube resting on the
floor 300 mm from the gripper. Every ``(predicate, name param)`` pair the
registry declares - 21 of them - accepted ``""`` and compiled clean. Twenty
pinned the term to a constant that reads as an honest miss: a bool predicate
``False``, a reward term ``0.0``. The twenty-first inverted it. ``gripper_prefix``
is compared with ``str.startswith``, and the empty string is the identity
prefix, so a blank one selects EVERY geom in the scene instead of the gripper's:

* ``grasped(body="cube", gripper_prefix="")`` answered ``True`` on the cube's
  contact with the floor it was placed on;
* ``run_policy(stop_when=...)`` returned ``status="success"``,
  ``stopped_reason="predicate"``, ``steps_used=1`` with the cube 0.000 mm from
  where it started;
* ``evaluate_benchmark`` over three episodes scored ``success_rate: 1.0``,
  ``n_success: 3``, ``avg_steps: 1.0`` under ``success_measured: true``.

That is the one violation in this module that reports a success which never
happened, and the arm-time probe cannot catch it: the collector feeding it
(``stop_when_referenced_entities``) gathers only non-empty strings, so a blank
name is never handed to ``can_resolve_body`` at all.

A non-string name is refused for the third reason the numeric domain gives - it
escaped as a bare ``TypeError: startswith first arg must be str or a tuple of
str, not int`` raised from inside the evaluation loop, naming neither the
predicate nor the clause it came from.

The sweeps derive their cases from ``PREDICATE_REGISTRY`` and the factories' own
annotations, so a predicate added later is covered by declaring its params;
there is no separate table to fall out of step with the registry. The counts
above are stated for the tree they were measured on and are not asserted.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from strands_robots.simulation.benchmark_spec import compile_stop_when
from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    _grasped,
    _is_prefix_param,
    make_predicate,
)

# The domain, restated in the test rather than imported: a REQUIRED name of
# something in the scene. ``str | None`` is deliberately absent - there ``None``
# is the documented sole-robot default of the ``base_*`` family, a value rather
# than a missing name - and ``test_the_pinned_domain_matches_the_module_rule``
# keeps the restatement from drifting.
_NAME_ANNOTATIONS = frozenset({"str"})


def _names_a_prefix(param: str) -> bool:
    """The prefix-kind rule, restated: a ``_prefix`` suffix.

    A local copy for the same reason ``_NAME_ANNOTATIONS`` is one, so this file
    states the rule it pins instead of asserting the implementation against
    itself.
    """
    return param.endswith("_prefix")


# One usable value per annotation the shipped factories declare on a param with
# no default, keyed on the WHOLE annotation. Exact rather than a substring test:
# ``"str" in "list[str]"`` is true, so a substring match would hand a container
# param a bare string, which ``name_list_error`` refuses.
_PROBE_VALUES: dict[str, Any] = {
    "float": 0.25,
    "int": 1,
    "str": "probe_name",
    "list[str]": ["probe_name"],
    "list[float]": [0.0, 0.0, 0.0],
}


def _buildable_kwargs(factory: Any) -> dict[str, Any]:
    """Kwargs that let *factory* be constructed, asserting on no value.

    Only params without a default need supplying, and each is supplied per its
    exact annotation. A name that resolves to nothing is fine here: every
    predicate already degrades to ``False`` for one, and these tests grade the
    refusal, not the verdict.
    """
    annotations = getattr(factory, "__annotations__", {})
    return {
        p.name: _PROBE_VALUES[str(annotations.get(p.name, ""))]
        for p in inspect.signature(factory).parameters.values()
        if p.default is inspect.Parameter.empty and str(annotations.get(p.name, "")) in _PROBE_VALUES
    }


def _name_params(factory: Any) -> list[str]:
    """Every param the factory annotates as a required entity name."""
    annotations = getattr(factory, "__annotations__", {})
    return [
        p.name
        for p in inspect.signature(factory).parameters.values()
        if str(annotations.get(p.name, "")) in _NAME_ANNOTATIONS
    ]


def _name_cases() -> list[tuple[str, str]]:
    """Every ``(predicate, entity-name param)`` pair the shipped registry declares."""
    return [(name, param) for name in sorted(PREDICATE_REGISTRY) for param in _name_params(PREDICATE_REGISTRY[name])]


# A registry that stopped declaring entity names would make every parametrized
# case below vacuous, so the sweep's own reach is asserted rather than assumed.
_MINIMUM_NAME_PARAMS = 15


class TestEveryDeclaredEntityNameCarriesTheDomain:
    def test_the_sweep_reaches_the_shipped_name_params(self):
        cases = _name_cases()
        assert len(cases) >= _MINIMUM_NAME_PARAMS, (
            f"the entity-name sweep found only {cases}; with fewer than "
            f"{_MINIMUM_NAME_PARAMS} pairs the parametrized cases below prove nothing"
        )
        # The prefix-matched one that read True, and one of each other kind.
        assert ("grasped", "gripper_prefix") in cases
        assert ("body_above_z", "body") in cases
        assert ("body_on", "body_a") in cases
        assert ("contact_between", "geom_a") in cases
        assert ("joint_above", "joint") in cases
        assert ("particles_inside", "container") in cases
        # Exactly one of them is matched as a prefix, so the two refusal
        # wordings below are each reached by the population.
        assert [c for c in cases if _names_a_prefix(c[1])] == [("grasped", "gripper_prefix")]

    @pytest.mark.parametrize(("name", "param"), _name_cases())
    def test_a_blank_name_is_refused_naming_the_predicate_and_the_param(self, name, param):
        kwargs = _buildable_kwargs(PREDICATE_REGISTRY[name])
        kwargs[param] = ""
        with pytest.raises(ValueError) as excinfo:
            make_predicate(name, **kwargs)
        message = str(excinfo.value)
        assert name in message, message
        assert param in message, message
        assert "non-empty string" in message, message

    @pytest.mark.parametrize(("name", "param"), _name_cases())
    def test_a_non_string_name_is_refused_before_the_evaluation_loop(self, name, param):
        kwargs = _buildable_kwargs(PREDICATE_REGISTRY[name])
        kwargs[param] = 42
        with pytest.raises(ValueError) as excinfo:
            make_predicate(name, **kwargs)
        message = str(excinfo.value)
        assert name in message, message
        assert param in message, message
        assert "must be a string" in message, message

    @pytest.mark.parametrize(("name", "param"), _name_cases())
    def test_a_usable_name_still_compiles(self, name, param):
        """The essence: the domain refuses what cannot name, not what can."""
        kwargs = _buildable_kwargs(PREDICATE_REGISTRY[name])
        kwargs[param] = "so101/gripper" if _names_a_prefix(param) else "cube"
        assert callable(make_predicate(name, **kwargs))

    def test_the_pinned_domain_matches_the_module_rule(self):
        """The restatements above and the module's own rule cannot drift apart."""
        for name, param in _name_cases():
            assert _names_a_prefix(param) is _is_prefix_param(param), (name, param)
        # A param the module would treat as a prefix but that the sweep does not
        # reach at all would leave the always-True wording unexercised.
        assert _is_prefix_param("gripper_prefix") is True
        assert _is_prefix_param("body") is False


class TestTheRefusalNamesTheConsequenceItPrevents:
    def test_a_blank_prefix_refusal_names_the_success_that_never_happened(self):
        with pytest.raises(ValueError) as excinfo:
            make_predicate("grasped", body="cube", gripper_prefix="")
        message = str(excinfo.value)
        assert "identity prefix" in message, message
        assert "EVERY geom" in message, message
        assert "step 1" in message, message

    def test_a_blank_body_refusal_names_the_constant_it_would_be_pinned_to(self):
        with pytest.raises(ValueError) as excinfo:
            make_predicate("body_above_z", body="", z=0.2)
        message = str(excinfo.value)
        assert "degrades to a constant" in message, message
        assert "False" in message, message

    def test_a_value_whose_repr_raises_is_described_rather_than_propagated(self):
        """The refusal is built through the shared renderer, so it cannot raise.

        ``repr`` of an int wider than :func:`sys.get_int_max_str_digits` raises
        ``ValueError``, and this guard runs only on the path whose purpose is to
        answer an unusable value with a message.
        """
        with pytest.raises(ValueError) as excinfo:
            make_predicate("body_above_z", body=10**5000, z=0.2)
        assert "bits" in str(excinfo.value), str(excinfo.value)

    def test_the_optional_robot_selector_keeps_its_none(self):
        """``str | None`` is a different domain: absence is a documented value."""
        assert callable(make_predicate("base_below_z", z=0.1, robot=None))
        assert callable(make_predicate("base_below_z", z=0.1))


class TestTheBlankPrefixReallySelectedEverything:
    """Non-vacuity: the refusal prevents a verdict, not a hypothetical."""

    @staticmethod
    def _sim_with_one_floor_contact() -> Any:
        """A sim reporting the cube resting on the ground, and nothing else."""

        class _FloorContactSim:
            def get_contacts(self) -> dict[str, Any]:
                return {
                    "status": "success",
                    "content": [
                        {"text": "1 contacts (1 touching)"},
                        {"json": {"contacts": [{"geom1": "cube_geom", "geom2": "ground", "active": True}]}},
                    ],
                }

        return _FloorContactSim()

    def test_a_blank_prefix_fires_on_the_floor_contact(self):
        """What the gate now refuses: the floor counted as the gripper."""
        assert _grasped("cube", "")(self._sim_with_one_floor_contact()) is True

    def test_a_named_gripper_prefix_does_not(self):
        assert _grasped("cube", "so101/gripper")(self._sim_with_one_floor_contact()) is False


class TestTheClauseSurfacesRefuseIt:
    def test_compile_stop_when_refuses_a_blank_prefix(self):
        with pytest.raises(ValueError, match="gripper_prefix"):
            compile_stop_when({"predicate": "grasped", "body": "cube", "gripper_prefix": ""})

    def test_compile_stop_when_refuses_a_blank_name_inside_a_group(self):
        with pytest.raises(ValueError, match="body"):
            compile_stop_when({"all": [{"predicate": "body_above_z", "body": "", "z": 0.2}]})

    def test_a_blank_name_inside_a_staged_reward_stage_is_refused(self):
        """The reason the domain lives at ``make_predicate``: stages call back into it."""
        with pytest.raises(ValueError, match="body_b"):
            make_predicate(
                "staged_reward",
                stages=[{"reward": {"predicate": "distance_neg", "body_a": "cube", "body_b": ""}}],
            )


class TestRunPolicyRefusesBeforeTheRollout:
    """The caller-visible envelope: a refusal instead of a one-step success."""

    def test_a_blank_prefix_returns_an_error_and_applies_no_action(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="entity_name_domain_test", mesh=False)
        try:
            sim.create_world()
            sim.add_robot(name="arm", data_config="so100")
            assert (
                sim.add_object(name="cube", shape="box", position=[0.3, 0.0, 0.03], size=[0.025] * 3)["status"]
                == "success"
            )
            result = sim.run_policy(
                robot_name="arm",
                policy_provider="mock",
                n_steps=50,
                control_frequency=30.0,
                fast_mode=True,
                stop_when={"predicate": "grasped", "body": "cube", "gripper_prefix": ""},
            )
        finally:
            sim.destroy()

        assert result["status"] == "error", result
        text = " ".join(block["text"] for block in result["content"] if "text" in block)
        assert "gripper_prefix" in text, text
        payload = next(block["json"] for block in result["content"] if "json" in block)
        assert payload["steps_used"] == 0, payload
        assert payload["stopped_reason"] == "error", payload
