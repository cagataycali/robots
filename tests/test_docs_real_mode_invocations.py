"""Every documented ``mode="real"`` invocation must name a real robot and real keywords.

``Robot(name, mode="real", **kwargs)`` is the one documented line that touches
physical hardware, and it is the line a reader copies verbatim. Two things about
it are decided at runtime rather than by the factory signature:

* **The name.** ``Robot()`` resolves it through the package registry, so a
  spelling that is neither a canonical name nor an alias raises ``ValueError``
  before anything is built.
* **The keywords.** ``mode="real"`` resolves the robot's ``hardware.lerobot_type``
  to a lerobot config dataclass, and a keyword is accepted only when that
  dataclass declares it or it appears in the cross-robot forwarding allowlist
  :data:`~strands_robots.hardware_robot._FORWARDABLE_KWARGS`. So the accepted set
  is a property of *the named robot*, not of the factory.

Neither is reachable from a signature. ``Robot`` ends in ``**kwargs: Any``, and
``tests/test_docs_python_examples_are_callable.py`` grades keywords against
signatures - its ``_accepted_keywords`` returns ``None`` (meaning "any keyword
binds") for a callee carrying ``**kwargs``. That is correct for its question and
it makes every ``Robot(...)`` keyword ungraded there, so the two modules are
complementary rather than overlapping: that one asks "would Python bind this
call", this one asks "would the runtime accept these values for this robot".

A block that documents a refusal is a negative example - it prints the exception
as its own output - so it is excluded by :func:`_documents_a_refusal` rather than
being graded as broken.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from pathlib import Path

import pytest

import strands_robots
import strands_robots.hardware_robot as hardware_robot
import strands_robots.robot as robot_factory
from strands_robots.registry import get_hardware_type, get_robot

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: A fence that grades nothing is indistinguishable from a clean sweep, so the
#: corpus size is asserted. The floor sits well below the current count; it only
#: has to fail if the extractor stops reaching the documentation.
_MINIMUM_GRADED_CALLS = 20


@dataclasses.dataclass(frozen=True)
class _Invocation:
    """One documented ``Robot(..., mode="real", ...)`` call.

    Attributes:
        location: ``path:line`` of the call, for a failure message that can be
            opened directly.
        name: The robot name as written in the documentation.
        keywords: Keyword names the call passes, excluding ``mode``.
    """

    location: str
    name: str
    keywords: tuple[str, ...]


def _documents_a_refusal(block: str) -> bool:
    """Return whether *block* prints an exception as its own output.

    A negative example shows the error the reader should expect - the leader-arm
    section documents that ``Robot()`` refuses every ``*_leader`` name by showing
    the ``ValueError``. Such a block is deliberately not runnable, so grading it
    would report the documentation's own teaching point as a defect.

    Args:
        block: The source text of one ``python`` fence.

    Returns:
        ``True`` when a comment line names an exception type.
    """
    return any(re.match(r"#\s*(\w*(?:Error|Exception))\b", line.strip()) for line in block.splitlines())


def _documented_real_mode_calls() -> list[_Invocation]:
    """Collect every documented ``Robot(..., mode="real", ...)`` call.

    Fences are parsed with :mod:`ast` rather than matched textually so a
    multi-line call and a keyword whose value itself contains a call are read
    correctly. A fence that is a fragment rather than a module does not parse and
    contributes nothing; the corpus floor is what stops that degrading silently.

    Returns:
        One :class:`_Invocation` per graded call, in file order.
    """
    found: list[_Invocation] = []
    sources = sorted((_REPO_ROOT / "docs").rglob("*.md")) + [_REPO_ROOT / "README.md"]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for fence in _PYTHON_FENCE.finditer(text):
            block = fence.group(1)
            if _documents_a_refusal(block):
                continue
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue
            fence_line = text[: fence.start()].count("\n") + 2
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                if not (isinstance(target, ast.Name) and target.id == "Robot"):
                    continue
                written = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                mode = written.get("mode")
                if not (isinstance(mode, ast.Constant) and mode.value == "real"):
                    continue
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    continue
                name = node.args[0].value
                if not isinstance(name, str):
                    continue
                found.append(
                    _Invocation(
                        location=f"{path.relative_to(_REPO_ROOT)}:{fence_line + node.lineno - 1}",
                        name=name,
                        keywords=tuple(k for k in written if k != "mode"),
                    )
                )
    return found


def _keywords_the_factory_owns() -> set[str]:
    """Return the keyword names accepted for every robot, whatever it is.

    Derived rather than listed, so a parameter added to either entry point is
    covered without editing this module: the sim/real factory's own parameters,
    the hardware wrapper's own parameters, and the cross-robot forwarding
    allowlist a robot's dataclass need not declare.

    Returns:
        The union of those three sets, without the binding-only names.
    """
    owned = (
        set(inspect.signature(robot_factory.Robot).parameters)
        | set(inspect.signature(hardware_robot.Robot.__init__).parameters)
        | set(hardware_robot._FORWARDABLE_KWARGS)
    )
    return owned - {"self", "kwargs", "name", "robot", "tool_name"}


def _keywords_the_robot_declares(name: str) -> set[str] | None:
    """Return the config fields declared for *name*, or ``None`` if unresolvable.

    Args:
        name: Robot name or alias as written in the documentation.

    Returns:
        The dataclass field names of the robot's lerobot config, or ``None``
        when the robot declares no lerobot type or lerobot does not register it -
        in which case the keywords are not graded rather than reported as wrong.
    """
    lerobot_type = get_hardware_type(name)
    if not lerobot_type:
        return None
    from lerobot.robots.config import RobotConfig

    hardware_robot._ensure_lerobot_robots_registered()
    config_cls = RobotConfig.get_known_choices().get(lerobot_type)
    if config_cls is None:
        return None
    return {field.name for field in dataclasses.fields(config_cls)}


def _names_no_registered_robot(name: str) -> bool:
    """Return whether ``Robot(name, ...)`` would refuse *name* as unknown.

    The one place the name rule lives, so the documentation sweep and the
    constructed exemplars below cannot drift apart.

    Args:
        name: Robot name or alias as written in the documentation.

    Returns:
        ``True`` when the registry resolves *name* to nothing, which is what
        makes ``Robot()`` raise before it builds anything.
    """
    return get_robot(name) is None


def _rejected_keywords(name: str, keywords: tuple[str, ...]) -> list[str]:
    """Return the keywords ``Robot(name, mode="real", ...)`` would refuse.

    The one place the acceptance rule lives, so the documentation sweep and the
    constructed exemplars below cannot drift apart.

    Args:
        name: Robot name or alias.
        keywords: Keyword names the call passes, excluding ``mode``.

    Returns:
        The rejected names, sorted. Empty when every keyword is accepted, and
        also empty when the robot's config cannot be resolved - an ungradable
        robot is not a wrong one.
    """
    declared = _keywords_the_robot_declares(name)
    if declared is None:
        return []
    return sorted(set(keywords) - (_keywords_the_factory_owns() | declared))


class TestTheCorpusIsReached:
    """Premises: without these, a clean sweep below would mean nothing."""

    def test_the_extractor_reaches_the_documentation(self) -> None:
        calls = _documented_real_mode_calls()
        assert len(calls) >= _MINIMUM_GRADED_CALLS, (
            f"only {len(calls)} documented mode='real' calls were found (expected at "
            f"least {_MINIMUM_GRADED_CALLS}); the extractor is no longer reaching the "
            "documentation, so a clean result would be meaningless"
        )

    def test_the_bimanual_recipe_is_among_them(self) -> None:
        """The multi-arm shape is graded, not just the single-``port`` majority."""
        calls = _documented_real_mode_calls()
        bimanual = [c for c in calls if "left_arm_config" in c.keywords]
        assert bimanual, "no documented mode='real' call passes a per-arm config"

    def test_the_factory_owns_a_nonempty_keyword_set(self) -> None:
        owned = _keywords_the_factory_owns()
        assert {"port", "cameras", "driver", "robot_ip"} <= owned


class TestEveryDocumentedRealModeCallNamesARegisteredRobot:
    """The name half - graded without lerobot, since the registry is enough."""

    def test_every_name_resolves(self) -> None:
        unknown = [
            f"{call.location}: Robot({call.name!r}, mode='real') - not a registered robot name or alias"
            for call in _documented_real_mode_calls()
            if _names_no_registered_robot(call.name)
        ]
        assert not unknown, "documented mode='real' calls naming no known robot:\n  " + "\n  ".join(unknown)


class TestEveryDocumentedRealModeKeywordIsAccepted:
    """The keyword half - needs lerobot to resolve the robot's config fields."""

    def test_every_keyword_is_accepted_by_the_named_robot(self) -> None:
        pytest.importorskip("lerobot.robots.config")
        offenders = []
        for call in _documented_real_mode_calls():
            rejected = _rejected_keywords(call.name, call.keywords)
            if rejected:
                offenders.append(
                    f"{call.location}: Robot({call.name!r}, mode='real') passes {rejected}, "
                    f"which neither the factory nor {call.name!r} accepts"
                )
        assert not offenders, "documented mode='real' calls that raise as written:\n  " + "\n  ".join(offenders)


class TestTheGraderReportsAPlantedMistake:
    """Non-vacuity: the rule must grade values, not the spelling of a fence."""

    def test_an_unregistered_name_is_reported(self) -> None:
        """The spelling the documentation used, and the one that replaced it."""
        assert _names_no_registered_robot("bi_so"), "'bi_so' became registered; this plant needs a new name"
        assert not _names_no_registered_robot("bi_so_follower"), "the corrected spelling must resolve"

    def test_a_leader_name_is_not_a_robot_name(self) -> None:
        """A teleoperator spelling must not pass the name rule either."""
        assert _names_no_registered_robot("so101_leader")
        assert not _names_no_registered_robot("so101")

    def test_the_name_rule_reaches_both_verdicts(self) -> None:
        outcomes = {_names_no_registered_robot(n) for n in ("bi_so", "so101_leader", "so101", "koch")}
        assert outcomes == {True, False}

    def test_a_per_arm_keyword_is_not_in_the_cross_robot_allowlist(self) -> None:
        """``left_port`` is accepted for no robot, which is why the old text raised."""
        owned = _keywords_the_factory_owns()
        assert "left_port" not in owned and "right_port" not in owned

    def test_the_bimanual_config_requires_a_per_arm_pair(self) -> None:
        pytest.importorskip("lerobot.robots.config")
        declared = _keywords_the_robot_declares("bi_so_follower")
        assert declared is not None
        assert {"left_arm_config", "right_arm_config"} <= declared
        assert "port" not in declared, "a bimanual config gained a single 'port'; the docs say it has none"


class TestTheSiblingGuardCannotSeeThis:
    """Why the existing signature-based grader is silent on these calls."""

    def test_the_factory_accepts_any_keyword_by_signature(self) -> None:
        for entry_point in (robot_factory.Robot, hardware_robot.Robot.__init__):
            parameters = inspect.signature(entry_point).parameters
            assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()), (
                f"{entry_point} lost its **kwargs; a signature-based grader can now see "
                "these calls and this module's complementarity claim needs revisiting"
            )

    def test_a_documented_refusal_is_not_graded(self) -> None:
        """The leader-arm section shows its ``ValueError``, so it is excluded."""
        assert _documents_a_refusal('Robot("so101_leader", mode="real")\n# ValueError: not a robot\n')
        assert not _documents_a_refusal('Robot("so101", mode="real", port="/dev/ttyACM0")\n')
        graded = {call.name for call in _documented_real_mode_calls()}
        assert "so101_leader" not in graded, "a documented refusal is being graded as a defect"


class TestTheKeywordRuleIsGradedOnConstructedExemplars:
    """The corpus is clean after the fix, so the rejection path needs exemplars.

    The old text is the flagged row: a correctly-named bimanual robot carrying
    the per-arm ``*_port`` spelling. Grading it here keeps the keyword half
    load-bearing without depending on a defect remaining in the documentation.
    """

    def test_the_old_bimanual_keywords_are_rejected_under_the_correct_name(self) -> None:
        pytest.importorskip("lerobot.robots.config")
        rejected = _rejected_keywords("bi_so_follower", ("left_port", "right_port"))
        assert rejected == ["left_port", "right_port"]

    def test_the_corrected_bimanual_keywords_are_accepted(self) -> None:
        pytest.importorskip("lerobot.robots.config")
        assert _rejected_keywords("bi_so_follower", ("left_arm_config", "right_arm_config")) == []

    def test_a_single_port_robot_accepts_port_and_refuses_a_per_arm_config(self) -> None:
        pytest.importorskip("lerobot.robots.config")
        assert _rejected_keywords("so101", ("port", "cameras")) == []
        assert _rejected_keywords("so101", ("left_arm_config",)) == ["left_arm_config"]

    def test_both_outcomes_occur_so_neither_branch_is_dead(self) -> None:
        pytest.importorskip("lerobot.robots.config")
        outcomes = {
            bool(_rejected_keywords(name, keywords))
            for name, keywords in (
                ("bi_so_follower", ("left_port",)),
                ("bi_so_follower", ("left_arm_config", "right_arm_config")),
                ("so101", ("port",)),
                ("so101", ("left_arm_config",)),
            )
        }
        assert outcomes == {True, False}
