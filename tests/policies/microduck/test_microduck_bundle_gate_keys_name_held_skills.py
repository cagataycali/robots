"""The velocity gate's two skill names are held to the membership ``active`` is.

:class:`~strands_robots.policies.microduck.MicroduckPolicyBundle` checks its
structural arguments carefully. It refuses an empty mapping; it refuses a value
that is not a :class:`MicroduckPolicy`, by name and by type; it refuses an
``active`` skill that is not one of its keys, naming the keys it does hold; and
:meth:`MicroduckPolicyBundle.switch` refuses an unknown name the same way. Its
one caller-supplied number, ``switch_on_velocity``, goes through the shared
``positive_finite_number_error`` domain.

``move_key`` and ``idle_key`` are skill names of exactly the kind ``active`` is,
and nothing checked them. They are read only by the velocity gate, and the gate
opens with::

    if self._move_key not in self._policies or self._idle_key not in self._policies:
        return

so a key that names no held skill did not fail - it made the gate inert. The
caller had opted into that gate by passing a threshold, and that threshold was
validated exhaustively, so the one argument that decided whether the gate could
ever fire was the one argument nobody graded.

The reachable case is the default. The bundle defaults to ``move_key="walk"`` /
``idle_key="stand"`` while Pollen ships its skills as ``alpha_walking``,
``alpha_stand``, ``roulade``, ``ball_kick_*`` - the names this module's own
docstring lists. A bundle keyed by the weight names it loads therefore
constructs, reports a validated threshold, and never switches: a biped commanded
to walk at 0.3 m/s stays on ``alpha_stand`` and every tick reports success. One
wrong key is enough, and it kills the whole gate rather than half of it - the
direction whose key *is* a held skill stops working too.

The check is scoped to the branch that reads the keys. With ``switch_on_velocity``
unset the gate never runs and the keys are never consulted, so refusing them then
would refuse a caller for a value the bundle does not look at - the rule
``serial_tool``, ``use_rtps``, ``use_rosbridge``, ``robot_mesh`` and
``_numeric_options`` all state for their own per-action options. The membership
is local rather than a shared domain for the same reason the two existing ones
are: it is a question about this instance's keys, not about the value's type.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from typing import Any

import pytest

from strands_robots.policies.microduck import MicroduckPolicy, MicroduckPolicyBundle
from strands_robots.policies.microduck import composite as composite_module
from tests.policies.microduck.test_microduck_policy import _obs_dict, _StubSession

#: The names Pollen ships its Microduck weights under, per this module's own
#: docstring - the keys a caller who loads ``alpha_walking.onnx`` reaches for.
SHIPPED_SKILLS = ("alpha_walking", "alpha_stand")

#: The bundle's defaults, which name no shipped weight.
DEFAULT_MOVE_KEY = "walk"
DEFAULT_IDLE_KEY = "stand"

#: A threshold the shared numeric domain accepts, so only the keys are in doubt.
USABLE_THRESHOLD = 0.1


def _policy() -> MicroduckPolicy:
    """A real policy over an injected stub session - no onnxruntime needed."""
    return MicroduckPolicy(session=_StubSession())


def _bundle(keys: tuple[str, ...], **kwargs: Any) -> MicroduckPolicyBundle:
    """A bundle holding one policy per name in ``keys``."""
    return MicroduckPolicyBundle({name: _policy() for name in keys}, **kwargs)


def _drive(bundle: MicroduckPolicyBundle, twist: list[float]) -> str:
    """Run one tick with ``target_velocity`` and report the active skill."""
    asyncio.run(bundle.get_actions(_obs_dict(), "", target_velocity=twist))
    return bundle.active


def _init_ast() -> ast.FunctionDef:
    """The bundle constructor's AST, read from the module source.

    Parsed from the module rather than from ``inspect.getsource`` of the method,
    whose text is indented as a class body and is not parseable on its own.
    """
    tree = ast.parse(inspect.getsource(composite_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    raise AssertionError("MicroduckPolicyBundle.__init__ not found in the module source")


def _gate_required_keys() -> list[str]:
    """Attribute names the gate requires to be held skills, read off its source.

    Derived from :meth:`MicroduckPolicyBundle._auto_switch` rather than listed,
    so a third gate key added later is held to the same construction-time check
    the day it lands instead of inheriting an exemption by being absent from a
    tuple here.
    """
    tree = ast.parse(inspect.getsource(composite_module))
    required: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_auto_switch":
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Compare) or not isinstance(sub.left, ast.Attribute):
                continue
            if not any(isinstance(op, ast.NotIn) for op in sub.ops):
                continue
            if any(ast.unparse(c) == "self._policies" for c in sub.comparators):
                required.append(sub.left.attr)
    return required


class TestAGateKeyMustNameAHeldSkill:
    """The refusals. Each of these constructed silently before.

    Every case pairs a threshold the shared domain accepts with a key that names
    no held skill, so the only thing in question is the key.
    """

    def test_a_bundle_keyed_by_the_shipped_weight_names_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names no held skill"):
            _bundle(SHIPPED_SKILLS, active="alpha_stand", switch_on_velocity=USABLE_THRESHOLD)

    @pytest.mark.parametrize("param", ["move_key", "idle_key"])
    def test_one_wrong_key_is_refused_even_when_the_other_is_held(self, param: str) -> None:
        with pytest.raises(ValueError, match="names no held skill"):
            _bundle(
                (DEFAULT_MOVE_KEY, DEFAULT_IDLE_KEY),
                active=DEFAULT_IDLE_KEY,
                switch_on_velocity=USABLE_THRESHOLD,
                **{param: "no-such-skill"},
            )

    @pytest.mark.parametrize("key", ["", "WALK", "walk ", "alpha_walking"])
    def test_a_near_miss_is_refused_rather_than_read_as_the_intended_skill(self, key: str) -> None:
        """Case, whitespace and the weight's own name are all not the key."""
        with pytest.raises(ValueError, match="names no held skill"):
            _bundle(
                (DEFAULT_MOVE_KEY, DEFAULT_IDLE_KEY),
                active=DEFAULT_IDLE_KEY,
                switch_on_velocity=USABLE_THRESHOLD,
                move_key=key,
            )

    def test_every_key_the_gate_requires_is_checked_at_construction(self) -> None:
        """Derived: each key ``_auto_switch`` requires is refused if unheld."""
        required = _gate_required_keys()
        assert required, "the gate's source names no key it requires to be a held skill"
        for attr in required:
            param = attr.lstrip("_")
            with pytest.raises(ValueError, match="names no held skill"):
                _bundle(
                    (DEFAULT_MOVE_KEY, DEFAULT_IDLE_KEY),
                    active=DEFAULT_IDLE_KEY,
                    switch_on_velocity=USABLE_THRESHOLD,
                    **{param: "no-such-skill"},
                )


class TestTheRefusalNamesWhatToFix:
    """A refusal has to name the parameter, its value, and the way out."""

    @staticmethod
    def _refusal(**kwargs: Any) -> str:
        with pytest.raises(ValueError) as caught:
            _bundle(SHIPPED_SKILLS, active="alpha_stand", switch_on_velocity=USABLE_THRESHOLD, **kwargs)
        return str(caught.value)

    def test_it_names_the_class(self) -> None:
        assert "MicroduckPolicyBundle" in self._refusal()

    def test_it_names_the_offending_parameter_and_its_value(self) -> None:
        text = self._refusal(move_key="no-such-skill")
        assert "move_key='no-such-skill'" in text

    def test_it_names_the_skills_the_bundle_does_hold(self) -> None:
        text = self._refusal()
        assert all(skill in text for skill in SHIPPED_SKILLS)

    def test_it_names_only_the_key_that_is_wrong(self) -> None:
        text = self._refusal(move_key="alpha_walking")
        assert "idle_key" in text and "move_key" not in text

    def test_it_names_both_keys_when_both_are_wrong(self) -> None:
        text = self._refusal()
        assert "move_key" in text and "idle_key" in text

    def test_it_names_the_gate_as_the_reason(self) -> None:
        assert "switch_on_velocity" in self._refusal()


class TestWhatStaysFirstClass:
    """Controls. Every one of these held before the guard and must still hold."""

    def test_the_gate_off_leaves_the_keys_unread_and_unchecked(self) -> None:
        bundle = _bundle(SHIPPED_SKILLS, active="alpha_stand")
        assert bundle.active == "alpha_stand"

    def test_an_explicit_switch_needs_no_gate_keys(self) -> None:
        bundle = _bundle(SHIPPED_SKILLS, active="alpha_stand")
        bundle.switch("alpha_walking")
        assert bundle.active == "alpha_walking"

    def test_the_documented_recipe_still_constructs_and_switches(self) -> None:
        bundle = _bundle(
            (DEFAULT_MOVE_KEY, DEFAULT_IDLE_KEY),
            active=DEFAULT_IDLE_KEY,
            switch_on_velocity=USABLE_THRESHOLD,
        )
        assert _drive(bundle, [0.3, 0.0, 0.0]) == DEFAULT_MOVE_KEY
        assert _drive(bundle, [0.0, 0.0, 0.0]) == DEFAULT_IDLE_KEY

    def test_the_shipped_weight_names_work_when_the_keys_name_them(self) -> None:
        bundle = _bundle(
            SHIPPED_SKILLS,
            active="alpha_stand",
            switch_on_velocity=USABLE_THRESHOLD,
            move_key="alpha_walking",
            idle_key="alpha_stand",
        )
        assert _drive(bundle, [0.3, 0.0, 0.0]) == "alpha_walking"
        assert _drive(bundle, [0.0, 0.0, 0.0]) == "alpha_stand"

    def test_one_skill_named_by_both_keys_is_a_legitimate_bundle(self) -> None:
        bundle = _bundle(("solo",), switch_on_velocity=USABLE_THRESHOLD, move_key="solo", idle_key="solo")
        assert _drive(bundle, [0.3, 0.0, 0.0]) == "solo"


class TestWhyTheGateWentInert:
    """Premises. These hold either way; they are why the guard belongs here."""

    def test_the_same_membership_on_active_is_refused_one_argument_over(self) -> None:
        with pytest.raises(ValueError, match="is not one of"):
            _bundle((DEFAULT_MOVE_KEY, DEFAULT_IDLE_KEY), active="stnad")

    def test_an_unknown_name_passed_to_switch_is_refused(self) -> None:
        bundle = _bundle((DEFAULT_MOVE_KEY, DEFAULT_IDLE_KEY), active=DEFAULT_IDLE_KEY)
        with pytest.raises(ValueError, match="unknown skill"):
            bundle.switch("stnad")

    def test_the_threshold_itself_is_held_to_the_shared_domain(self) -> None:
        with pytest.raises(ValueError, match="switch_on_velocity"):
            _bundle((DEFAULT_MOVE_KEY, DEFAULT_IDLE_KEY), switch_on_velocity=0.0)

    def test_the_defaults_name_no_shipped_weight(self) -> None:
        """The premise that makes the default the reachable case."""
        assert DEFAULT_MOVE_KEY not in SHIPPED_SKILLS
        assert DEFAULT_IDLE_KEY not in SHIPPED_SKILLS

    def test_the_gate_returns_early_rather_than_raising_for_an_unheld_key(self) -> None:
        """The mechanism: inert, not loud, which is why nothing surfaced it.

        Reached by bypassing the constructor, which now makes the state
        unreachable - so this also pins that the gate's own guard still absorbs
        it rather than raising a ``KeyError`` from ``self._policies[...]``.
        """
        bundle = _bundle((DEFAULT_MOVE_KEY, DEFAULT_IDLE_KEY), active=DEFAULT_IDLE_KEY)
        bundle._switch_on_velocity = USABLE_THRESHOLD
        bundle._move_key = "no-such-skill"
        assert _drive(bundle, [0.3, 0.0, 0.0]) == DEFAULT_IDLE_KEY


class TestTheCheckIsScopedToTheGate:
    """Structural: the membership sits inside the branch that reads the keys."""

    def test_the_check_is_nested_under_the_threshold_being_set(self) -> None:
        tree = _init_ast()
        gate_branches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and "_switch_on_velocity is not None" in ast.unparse(node.test)
        ]
        assert gate_branches, "no branch guards on the threshold being set"
        assert any("names no held skill" in ast.unparse(branch) for branch in gate_branches), (
            "the gate-key membership check is not nested under the threshold branch, so a "
            "caller with the gate off would be refused for keys the bundle never reads"
        )
