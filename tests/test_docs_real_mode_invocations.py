"""Every documented ``mode="real"`` invocation must name a real robot and real keywords.

``Robot(name, mode="real", **kwargs)`` is the one documented line that touches
physical hardware, and it is the line a reader copies verbatim. Two things about
it are decided at runtime rather than by the factory signature:

* **The name.** ``Robot()`` resolves it through the package registry, so a
  spelling that is neither a canonical name nor an alias raises ``ValueError``
  before anything is built.
* **The keywords.** Which surface they must satisfy depends on the driver the
  call resolves to, so the accepted set is a property of *the named robot and
  its driver*, not of the factory.

  * ``driver="lerobot"`` (today's default) resolves the robot's
    ``hardware.lerobot_type`` to a lerobot config dataclass and forwards the
    keywords into it, so one the dataclass does not declare raises - unless it
    appears in the cross-robot forwarding allowlist
    :data:`~strands_robots.hardware_robot._FORWARDABLE_KWARGS`. That is the half
    graded here.
  * ``driver="strands"`` never reaches a lerobot config at all.
    :func:`~strands_robots.robot._build_native_driver` forwards the keywords
    verbatim to the registered driver's ``__init__``, and every shipped driver
    ends in ``**kwargs`` which it pops its own keywords out of - the Feetech
    driver's ``port`` / ``transport``, the UR driver's ``rtde_frequency`` - and
    keeps the remainder in ``_extras``. So *any* keyword binds and none is
    refused: there is no acceptance rule to grade, and grading such a call
    against the lerobot dataclass reports a working documented line as broken.
    Those calls are therefore excluded from the keyword half by
    :func:`_builds_a_native_driver`, on the same reasoning that makes the
    signature-based sibling return ``None`` for a callee carrying ``**kwargs``.

Which driver a call resolves to is not the spelling of its ``driver=`` either -
an absent keyword defers to the robot's registry ``hardware.driver`` and then to
:data:`~strands_robots.registry.DEFAULT_DRIVER`. :func:`resolve_driver` is that
rule, so it is called rather than re-implemented here.

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
from strands_robots.drivers import driver_choice_error, resolve_driver
from strands_robots.registry import get_hardware_type, get_robot

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: A fence that grades nothing is indistinguishable from a clean sweep, so the
#: corpus size is asserted. The floor sits well below the current count; it only
#: has to fail if the extractor stops reaching the documentation.
_MINIMUM_GRADED_CALLS = 20

#: The same premise for the keyword half alone, which grades only the calls a
#: lerobot config backs. Excluding the native-driver calls must leave a corpus,
#: not empty the check: 24 of the 42 documented calls resolve to lerobot today.
_MINIMUM_LEROBOT_CALLS = 10


@dataclasses.dataclass(frozen=True)
class _Invocation:
    """One documented ``Robot(..., mode="real", ...)`` call.

    Attributes:
        location: ``path:line`` of the call, for a failure message that can be
            opened directly.
        name: The robot name as written in the documentation.
        keywords: Keyword names the call passes, excluding ``mode``.
        driver: The ``driver=`` value as written, or ``None`` when the call
            passes none (deferring to the registry) or passes a computed one.
    """

    location: str
    name: str
    keywords: tuple[str, ...]
    driver: str | None = None


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
                driver = written.get("driver")
                found.append(
                    _Invocation(
                        location=f"{path.relative_to(_REPO_ROOT)}:{fence_line + node.lineno - 1}",
                        name=name,
                        keywords=tuple(k for k in written if k != "mode"),
                        driver=driver.value
                        if isinstance(driver, ast.Constant) and isinstance(driver.value, str)
                        else None,
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


def _refused_driver_choice(driver: str | None) -> str | None:
    """Return why ``driver=`` itself is refused, or ``None`` when it is a choice.

    Read before the driver is resolved, because an unknown spelling makes
    ``Robot()`` raise on the keyword rather than build anything - and because
    :func:`resolve_driver` raises on it, which would turn a documentation defect
    into an error inside the sweep instead of a reported location.

    Args:
        driver: The ``driver=`` value as written, or ``None`` when unset.

    Returns:
        The refusal reason, or ``None``.
    """
    if driver is None:
        return None
    return driver_choice_error(driver, "driver", "Robot")


def _builds_a_native_driver(name: str, driver: str | None) -> bool:
    """Return whether this call is built by a native driver rather than lerobot.

    :func:`~strands_robots.drivers.resolve_driver` is the rule, not the spelling
    of ``driver=``: an absent keyword defers to the robot's registry
    ``hardware.driver`` and then to the package default, so a robot that
    declares a native driver takes the native path with no keyword at all.

    Args:
        name: Robot name or alias as written in the documentation.
        driver: The ``driver=`` value as written, or ``None`` when unset. Must
            already be known to be a valid choice.

    Returns:
        ``True`` when the keywords are forwarded to a driver's ``**kwargs``
        instead of into a lerobot config dataclass.
    """
    return resolve_driver(name, driver) != "lerobot"


def _rejected_keywords(name: str, keywords: tuple[str, ...], driver: str | None = None) -> list[str]:
    """Return the keywords ``Robot(name, mode="real", ...)`` would refuse.

    The one place the acceptance rule lives, so the documentation sweep and the
    constructed exemplars below cannot drift apart.

    Args:
        name: Robot name or alias.
        keywords: Keyword names the call passes, excluding ``mode``.
        driver: The call's ``driver=`` as written, or ``None`` when unset.

    Returns:
        The rejected names, sorted. Empty when every keyword is accepted, and
        also empty when the call cannot be graded rather than being wrong: a
        native driver takes every keyword through ``**kwargs`` and refuses
        none, and a robot whose lerobot config will not resolve declares no
        field set to check against.
    """
    if _builds_a_native_driver(name, driver):
        return []
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

    def test_the_keyword_half_still_grades_a_corpus_of_lerobot_calls(self) -> None:
        """Excluding the native-driver calls must not empty the keyword half."""
        pytest.importorskip("lerobot.robots.config")
        graded = [
            call
            for call in _documented_real_mode_calls()
            if _refused_driver_choice(call.driver) is None and not _builds_a_native_driver(call.name, call.driver)
        ]
        assert len(graded) >= _MINIMUM_LEROBOT_CALLS, (
            f"only {len(graded)} documented mode='real' calls are backed by a lerobot config "
            f"(expected at least {_MINIMUM_LEROBOT_CALLS}); the keyword half now grades almost "
            "nothing, so a clean result would be meaningless"
        )

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
            refused = _refused_driver_choice(call.driver)
            if refused is not None:
                offenders.append(f"{call.location}: {refused}")
                continue
            rejected = _rejected_keywords(call.name, call.keywords, call.driver)
            if rejected:
                offenders.append(
                    f"{call.location}: Robot({call.name!r}, mode='real') passes {rejected}, "
                    f"which neither the factory nor the lerobot config for {call.name!r} accepts"
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

    def test_a_native_driver_keyword_is_graded_against_the_driver_not_the_config(self) -> None:
        """``transport`` is the Feetech driver's own keyword and no lerobot field.

        The pair is the whole point: the same keyword on the same robot is
        accepted on the native path and refused on the lerobot one, so the rule
        reads the driver rather than the robot alone.
        """
        pytest.importorskip("lerobot.robots.config")
        declared = _keywords_the_robot_declares("so101")
        assert declared is not None and "transport" not in declared
        assert _rejected_keywords("so101", ("transport",), "strands") == []
        assert _rejected_keywords("so101", ("transport",), "lerobot") == ["transport"]

    def test_the_native_path_is_decided_by_resolution_not_the_written_keyword(self) -> None:
        """A robot declaring a native driver takes that path with no keyword."""
        assert _builds_a_native_driver("so101", "strands")
        assert not _builds_a_native_driver("so101", "lerobot")
        assert not _builds_a_native_driver("so101", None), "the SO arms declare no native driver yet"
        assert _builds_a_native_driver("yahboom_m3pro", None), "a declared native driver needs no keyword"

    def test_an_unknown_driver_spelling_is_reported_rather_than_raising(self) -> None:
        """A near-miss spelling is a documentation defect with a location."""
        assert _refused_driver_choice("strand") is not None
        assert _refused_driver_choice("strands") is None
        assert _refused_driver_choice(None) is None

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
