"""A wrong or missing predicate keyword is refused by name, on every DSL surface.

docs/learn/simulation/predicates-and-rollouts.md promises that a wrong keyword
is a ``ValueError`` naming the accepted ones. Before this test the factory's own
``TypeError`` (``_contact_between() got an unexpected keyword argument
'body_a'``) reached ``make_predicate`` callers, ``run_policy(stop_when=...)``,
``eval_policy(success_when=...)`` and benchmark files verbatim, and it names no
accepted keyword.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation import predicates
from strands_robots.simulation.benchmark_spec import compile_stop_when
from strands_robots.simulation.predicates import make_predicate


class TestAWrongKeywordNamesTheAcceptedOnes:
    def test_make_predicate_refuses_a_wrong_keyword_with_the_accepted_list(self) -> None:
        with pytest.raises(ValueError) as exc:
            make_predicate("contact_between", body_a="cube", body_b="table")
        text = str(exc.value)
        assert "contact_between" in text
        assert "geom_a, geom_b" in text
        assert "body_a" in text
        assert "unexpected keyword" not in text

    def test_make_predicate_refuses_a_missing_required_keyword_by_name(self) -> None:
        with pytest.raises(ValueError) as exc:
            make_predicate("grasped", body="cube")
        text = str(exc.value)
        assert "missing gripper_prefix" in text
        assert "body, gripper_prefix" in text

    def test_the_stop_when_surface_carries_the_same_refusal(self) -> None:
        with pytest.raises(ValueError) as exc:
            compile_stop_when({"predicate": "contact_between", "body_a": "cube", "body_b": "table"})
        text = str(exc.value)
        assert "stop_when" in text
        assert "geom_a, geom_b" in text
        assert "unexpected keyword" not in text

    def test_a_well_formed_call_still_compiles(self) -> None:
        pred = make_predicate("contact_between", geom_a="cube", geom_b="table")
        assert callable(pred)

    def test_a_factory_that_takes_any_keyword_is_not_second_guessed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def factory(**kw: object):
            return lambda _sim: True

        monkeypatch.setitem(predicates.PREDICATE_REGISTRY, "anything_goes", factory)
        assert callable(make_predicate("anything_goes", whatever=1))

    def test_every_registered_factory_has_a_readable_signature(self) -> None:
        # The refusal is built from the signature, so every shipped predicate must expose one
        # with at least one keyword; otherwise the accepted list would be empty text.
        for name, factory in predicates.PREDICATE_REGISTRY.items():
            err = predicates._keyword_set_error(name, factory, {"definitely_not_a_keyword": 1})
            assert err is not None and name in err and "definitely_not_a_keyword" in err, name
