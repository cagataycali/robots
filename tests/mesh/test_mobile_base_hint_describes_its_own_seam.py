"""A refusal's hint must describe the grammar that produced it.

``MobileBaseRobot`` validates two independently-overridable seams: ``node_name``
against ``_NAME_RE`` and every topic/service against ``_TOPIC_RE``. Both
refusals used to append one shared sentence, ``_NAME_HINT``. On the single
platform whose two grammars diverge - ``RtpsRobot``, where a topic is written to
DDS directly and must be absolute while a ``node_name`` may be relative or
``~``-prefixed - no one sentence is true of both seams, so the platform's only
escape was to silence the hint entirely (``_NAME_HINT = ""``). The result was
that the strictest grammar in the tree was the one that said nothing about
itself: ``RtpsRobot("tb", "cmd_vel")`` refused with a bare
``invalid cmd_vel_topic: 'cmd_vel'``.

Why the existing suite was silent on it: ``test_mobile_base.py``'s
``test_shipped_classes_keep_their_own_name_grammar`` pins the divergence
behaviourally, and ``test_rtps_robot.py`` pins the refusal, but both match on
``"invalid cmd_vel_topic"`` - the prefix, which stops immediately before the
hint. The grammar rule itself was written down twice in prose (that test's
docstring, and the comment above ``_RTPS_TOPIC_RE``) and never asserted.

The seam table below is read out of the base's own ``_check_*`` methods rather
than listed, so a third seam added later is held to the same rule on arrival.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest

import strands_robots.mesh as mesh_pkg
from strands_robots.mesh import MobileBaseRobot, RosBridgedRobot, RtpsRobot

#: Extracts the example value a hint offers ("... like /turtle1/cmd_vel)").
_EXAMPLE_RE = re.compile(r"\blike\s+(\S+)")


def _seams() -> tuple[tuple[str, str, str | None], ...]:
    """Every seam the base validates, as ``(method, pattern_attr, hint_attr)``.

    Derived from the bodies of ``MobileBaseRobot._check_*`` so the table cannot
    drift from the code: a seam is whatever forwards a class-level pattern to
    ``_check``. ``hint_attr`` is ``None`` for a seam that forwards no hint,
    which is itself a finding rather than an error here.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(MobileBaseRobot)))
    cls_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    seams: list[tuple[str, str, str | None]] = []
    for fn in cls_node.body:
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("_check_"):
            continue
        call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("._check"))
        attrs = [ast.unparse(a).removeprefix("cls.") for a in call.args if ast.unparse(a).startswith("cls.")]
        patterns = [a for a in attrs if a.endswith("_RE")]
        hints = [a for a in attrs if a.endswith("_HINT")]
        assert len(patterns) == 1, f"{fn.name} forwards {patterns} patterns, expected exactly one"
        seams.append((fn.name, patterns[0], hints[0] if hints else None))
    return tuple(seams)


def _platforms() -> tuple[type[MobileBaseRobot], ...]:
    """The base plus every ``MobileBaseRobot`` subclass the package declares.

    Discovered by walking the public modules of ``strands_robots.mesh`` rather
    than listed, so a platform added later is graded on arrival. Keyed on
    ``__module__`` so a class is attributed to the module that defines it.
    """
    root = Path(mesh_pkg.__file__).parent
    found: dict[str, type[MobileBaseRobot]] = {MobileBaseRobot.__name__: MobileBaseRobot}
    for path in sorted(root.glob("*.py")):
        if path.stem.startswith("__"):
            continue
        module = __import__(f"strands_robots.mesh.{path.stem}", fromlist=["_"])
        for obj in vars(module).values():
            if inspect.isclass(obj) and issubclass(obj, MobileBaseRobot) and obj.__module__ == module.__name__:
                found[obj.__name__] = obj
    return tuple(found[name] for name in sorted(found))


def _diverges(platform: type[MobileBaseRobot]) -> bool:
    """Whether this platform's two seams use different grammars."""
    return platform._NAME_RE.pattern != platform._TOPIC_RE.pattern


_PLATFORM_SEAM = [(platform, seam) for platform in _platforms() for seam in _seams()]


def _ids(rows: list[Any]) -> list[str]:
    return [f"{p.__name__}-{s[0]}" for p, s in rows]


class TestThePopulationIsDerivedAndBothBranchesAreReached:
    """Premises. Without these the rules below could pass by finding nothing."""

    def test_the_discovery_finds_every_shipped_platform(self) -> None:
        names = {p.__name__ for p in _platforms()}
        assert {"MobileBaseRobot", "RosBridgedRobot", "RtpsRobot"} <= names

    def test_the_seam_table_finds_both_seams(self) -> None:
        methods = {seam[0] for seam in _seams()}
        assert {"_check_name", "_check_topic"} <= methods

    def test_both_sides_of_the_divergence_split_are_populated(self) -> None:
        """A platform-wide rule graded over a uniform population is vacuous."""
        diverging = [p.__name__ for p in _platforms() if _diverges(p)]
        coinciding = [p.__name__ for p in _platforms() if not _diverges(p)]
        assert diverging, "no platform's seams diverge, so the two-hint rule is untested"
        assert coinciding, "no platform's seams coincide, so the shared-hint case is untested"

    def test_the_rtps_topic_grammar_really_is_the_stricter_one(self) -> None:
        """The premise the hint states: values rclpy resolves, DDS cannot."""
        for value in ("cmd_vel", "~/cmd_vel", "/cmd_vel/"):
            assert RosBridgedRobot._TOPIC_RE.match(value), value
            assert not RtpsRobot._TOPIC_RE.match(value), value


class TestEverySeamCarriesAHintForItsOwnGrammar:
    """The structural rule, derived over every (platform, seam) pair."""

    def test_every_seam_forwards_a_hint_of_its_own(self) -> None:
        borrowed = [(m, h) for m, _, h in _seams() if h is None]
        assert not borrowed, f"seams forwarding no hint of their own: {borrowed}"

    def test_no_two_seams_share_one_hint_attribute(self) -> None:
        """Sharing one constant is what forced a divergent platform to go silent."""
        hints = [h for _, _, h in _seams()]
        assert len(set(hints)) == len(hints), f"seams share a hint attribute: {hints}"

    @pytest.mark.parametrize(("platform", "seam"), _PLATFORM_SEAM, ids=_ids(_PLATFORM_SEAM))
    def test_the_hint_is_not_empty(self, platform: type[MobileBaseRobot], seam: tuple[str, str, str | None]) -> None:
        _, _, hint_attr = seam
        assert hint_attr is not None
        assert getattr(platform, hint_attr).strip(), (
            f"{platform.__name__}.{hint_attr} is empty, so its refusal says nothing about what a good value looks like"
        )

    @pytest.mark.parametrize(("platform", "seam"), _PLATFORM_SEAM, ids=_ids(_PLATFORM_SEAM))
    def test_the_hint_offers_an_example_its_own_pattern_accepts(
        self, platform: type[MobileBaseRobot], seam: tuple[str, str, str | None]
    ) -> None:
        """A hint pointing at a value its own seam refuses is worse than none."""
        _, pattern_attr, hint_attr = seam
        assert hint_attr is not None
        hint = getattr(platform, hint_attr)
        match = _EXAMPLE_RE.search(hint)
        assert match, f"{platform.__name__}.{hint_attr} offers no 'like <example>'"
        example = match.group(1).rstrip(").,;")
        pattern = getattr(platform, pattern_attr)
        assert pattern.match(example), (
            f"{platform.__name__}.{hint_attr} offers {example!r}, which its own "
            f"{pattern_attr} ({pattern.pattern}) refuses"
        )


class TestADivergentPlatformSaysSoTwice:
    """A platform whose grammars differ must not describe them with one sentence."""

    @pytest.mark.parametrize("platform", [p for p in _platforms() if _diverges(p)], ids=lambda p: p.__name__)
    def test_its_two_hints_differ(self, platform: type[MobileBaseRobot]) -> None:
        assert platform._NAME_HINT != platform._TOPIC_HINT, (
            f"{platform.__name__} uses different grammars for its two seams but describes both with the same sentence"
        )


class TestTheRtpsTopicRefusalNamesTheAbsoluteRule:
    """The regression. Each value rclpy resolves and DDS cannot must say why."""

    @pytest.mark.parametrize("topic", ["cmd_vel", "~/cmd_vel", "/cmd_vel/"])
    def test_the_refusal_names_the_absolute_requirement(self, topic: str) -> None:
        with pytest.raises(ValueError, match="absolute topic") as caught:
            RtpsRobot("tb", topic)
        assert topic in str(caught.value)

    @pytest.mark.parametrize("topic", ["cmd_vel", "~/cmd_vel", "/cmd_vel/"])
    def test_the_refusal_offers_an_accepted_example(self, topic: str) -> None:
        """The example must be a value this class really takes, not a shape."""
        with pytest.raises(ValueError) as caught:
            RtpsRobot("tb", topic)
        match = _EXAMPLE_RE.search(str(caught.value))
        assert match, str(caught.value)
        RtpsRobot("tb", match.group(1).rstrip(").,;"))

    def test_the_refusal_says_why_a_relative_name_cannot_work(self) -> None:
        with pytest.raises(ValueError, match="use_rtps"):
            RtpsRobot("tb", "cmd_vel")


class TestASharedGrammarKeepsASharedSentence:
    """A seam whose grammar is inherited must inherit the sentence too.

    ``RtpsRobot`` silenced ``_NAME_HINT`` to escape the shared-constant problem,
    which cost the ``node_name`` seam a hint it had no reason to lose: that seam
    uses the base's grammar verbatim.
    """

    def test_the_rtps_name_seam_regains_the_inherited_sentence(self) -> None:
        assert RtpsRobot._NAME_HINT == MobileBaseRobot._NAME_HINT
        with pytest.raises(ValueError, match=re.escape("expected a graph name")):
            RtpsRobot("bad name", "/cmd_vel")

    def test_a_platform_whose_seams_coincide_describes_them_identically(self) -> None:
        """The ROS 2 bridge overrides one grammar for both seams, so one sentence."""
        assert RosBridgedRobot._NAME_RE.pattern == RosBridgedRobot._TOPIC_RE.pattern
        assert RosBridgedRobot._NAME_HINT == RosBridgedRobot._TOPIC_HINT


class TestNoOverReach:
    """Every expectation here is one the pre-fix code also met."""

    def test_a_valid_rtps_topic_is_still_accepted(self) -> None:
        assert RtpsRobot("tb", "/turtle1/cmd_vel").cmd_vel_topic == "/turtle1/cmd_vel"

    def test_the_rtps_name_seam_shares_the_base_grammar_and_still_refuses(self) -> None:
        """The grammar is unchanged; only what the refusal says about it moves."""
        assert RtpsRobot._NAME_RE.pattern == MobileBaseRobot._NAME_RE.pattern
        with pytest.raises(ValueError, match=re.escape("invalid node_name: 'bad name'")):
            RtpsRobot("bad name", "/cmd_vel")

    def test_the_ros_2_bridge_messages_are_unchanged(self) -> None:
        """Both seams named the ROS 2 grammar before this change and still do."""
        assert RosBridgedRobot._NAME_HINT == " (expected a ROS 2 graph name like /turtle1/cmd_vel)"
        with pytest.raises(ValueError, match=re.escape("ROS 2 graph name")):
            RosBridgedRobot("bad name", "/cmd_vel", "/odom")
        with pytest.raises(ValueError, match=re.escape("ROS 2 graph name")):
            RosBridgedRobot("tb", "bad topic", "/odom")

    def test_the_ros_2_bridge_still_accepts_a_relative_topic(self) -> None:
        """rclpy resolves it, so widening the hint must not narrow the grammar."""
        assert RosBridgedRobot("tb", "relative/cmd_vel", "/odom").cmd_vel_topic == "relative/cmd_vel"

    def test_the_refusal_still_quotes_the_label_and_the_value(self) -> None:
        with pytest.raises(ValueError, match=re.escape("invalid cmd_vel_topic: 'cmd_vel'")):
            RtpsRobot("tb", "cmd_vel")


def _modules_redefining_the_shared_check(source_by_module: dict[str, str]) -> list[str]:
    """Modules defining a ``_check`` other than the one the base owns.

    A second copy is how the hint gets dropped: ``rtps_robot`` shipped a
    module-level ``_check`` with no caller whose refusal omitted the hint
    entirely, one edit away from being adopted.
    """
    offenders = []
    for name, source in source_by_module.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == "_check":
                offenders.append(name)
    return sorted(set(offenders))


class TestOnlyTheBaseOwnsTheSharedCheck:
    """One ``_check`` means one place a hint can be dropped."""

    def test_no_other_mesh_module_defines_a_check(self) -> None:
        root = Path(mesh_pkg.__file__).parent
        sources = {
            path.stem: path.read_text(encoding="utf-8")
            for path in sorted(root.glob("*.py"))
            if path.stem not in {"__init__", "_mobile_base"}
        }
        assert sources, "found no mesh modules to scan"
        assert _modules_redefining_the_shared_check(sources) == []

    def test_the_scan_flags_a_constructed_second_copy(self) -> None:
        """Non-vacuity: the rule above must be able to fail."""
        planted = "def _check(label, value, pattern):\n    return value\n"
        assert _modules_redefining_the_shared_check({"planted": planted}) == ["planted"]
