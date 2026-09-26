"""The showcase that claims every ``use_ros`` action must call every one of them.

``examples/ros2/use_ros/showcase.py`` is the one place in the tree where the
tool's whole vocabulary is driven against a live ROS 2 graph, and four surfaces
sell it as complete - the script's own docstring, the example README ("It
exercises every action"), ``docker-compose.yml`` and
``docs/ros2-integration.md`` ("drives a real ``turtlesim`` through every
``use_ros`` action"). Its captured ``sample_output.txt`` is then the evidence a
reader trusts for what each verb returns.

Completeness is exactly what no other guard grades.
``test_docs_tool_action_values_are_dispatched`` reports an action the prose names
that no tool dispatches, and says so: "the rule is one-directional ... only an
action no tool dispatches is" reported. So a verb the showcase never calls is
invisible there, which is how the two action-client verbs - ``list_actions`` and
``action_send_goal``, the only ones whose reply is a terminal status and whose
timeout cancels rather than abandons - went undriven under a claim of every
action, on a graph (turtlesim) that publishes an action server.

Both rosters are read from the tree rather than listed here: the vocabulary from
the tool's published ``action`` argument, the coverage from the ``action=``
keywords of the calls the script actually makes. Prose is deliberately not
credited - the docstring on the pre-fix script named all ten verbs while calling
eight - so a comment or a docstring cannot satisfy this rule.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots
from strands_robots.tools.use_ros import use_ros

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_SHOWCASE = _REPO_ROOT / "examples" / "ros2" / "use_ros" / "showcase.py"
# The two modules that dispatch on the value: the tool routes each verb to the
# transport, the transport answers it.
_DISPATCHERS = (
    _REPO_ROOT / "strands_robots" / "tools" / "use_ros.py",
    _REPO_ROOT / "strands_robots" / "ros.py",
)

# The tool publishes its vocabulary in the ``action`` argument of its docstring:
# ``action: One of ``status``, ``list_topics``, ... ``action_send_goal``.``
_PUBLISHED = re.compile(r"action:\s*One of(.+?)\.\n", re.DOTALL)
_BACKTICKED = re.compile(r"``([a-z_][a-z0-9_]*)``")

# A roster that read short would accept a showcase covering almost nothing.
_MINIMUM_PUBLISHED = 8


def _published_actions() -> set[str]:
    """Every action value the ``use_ros`` tool publishes to its callers."""
    doc = use_ros.__doc__ or ""
    match = _PUBLISHED.search(doc)
    assert match, "the use_ros docstring no longer publishes an action vocabulary"
    return set(_BACKTICKED.findall(match.group(1)))


def _dispatched_actions() -> set[str]:
    """Every action value the tool or its transport compares against."""
    found: set[str] = set()
    for path in _DISPATCHERS:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left, *node.comparators]
            if not any(isinstance(o, ast.Name) and o.id == "action" for o in operands):
                continue
            for operand in operands:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                    found.add(operand.value)
                if isinstance(operand, ast.Tuple | ast.List | ast.Set):
                    found |= {e.value for e in operand.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return found


def _called_actions(source: str) -> set[str]:
    """The action values ``source`` passes at a call site.

    Only a string literal passed as the ``action`` keyword of a call counts. A
    verb named in prose - a docstring, a comment, a printed line - is not a
    call, and crediting one would let the claim be satisfied by restating it.
    """
    called: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "action" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    called.add(keyword.value.value)
    return called


class TestTheShowcaseCoversThePublishedVocabulary:
    """Every published verb reaches the live graph in the showcase."""

    def test_no_published_action_goes_undriven(self) -> None:
        """A verb the showcase never calls is a gap in its own claim."""
        published = _published_actions()
        called = _called_actions(_SHOWCASE.read_text(encoding="utf-8"))
        missing = sorted(published - called)
        assert not missing, (
            f"{_SHOWCASE.relative_to(_REPO_ROOT)} claims every use_ros action but never calls: "
            f"{missing} (called: {sorted(called)})"
        )

    def test_the_rosters_are_read(self) -> None:
        """A clean result must mean both rosters were read, not that neither was."""
        published = _published_actions()
        assert len(published) >= _MINIMUM_PUBLISHED, f"published vocabulary read as {sorted(published)}"
        assert {"status", "publish", "action_send_goal"} <= published
        assert _called_actions(_SHOWCASE.read_text(encoding="utf-8"))

    def test_every_published_action_is_a_dispatched_verb(self) -> None:
        """The published roster names verbs the code answers, not fiction.

        Without this the rule could be satisfied by a documented vocabulary
        that shrank: a verb dropped from the docstring stops being required of
        the showcase, and nothing would report that it is still dispatched.
        """
        published = _published_actions()
        dispatched = _dispatched_actions()
        assert published <= dispatched, f"published but not dispatched: {sorted(published - dispatched)}"
        assert dispatched <= published, f"dispatched but not published: {sorted(dispatched - published)}"


class TestTheGraderIsLoadBearing:
    """The rule reports a planted omission, and only an omission."""

    def test_a_planted_omission_is_reported(self) -> None:
        """A script skipping one verb must be short of the vocabulary."""
        planted = "use_ros(action='status')\nuse_ros(action='list_topics')\n"
        assert "echo" not in _called_actions(planted)

    def test_a_planted_full_sweep_is_accepted(self) -> None:
        """A script calling the whole vocabulary must satisfy the rule."""
        published = _published_actions()
        planted = "".join(f"run(action={a!r})\n" for a in sorted(published))
        assert not published - _called_actions(planted)

    def test_prose_naming_a_verb_is_not_credited(self) -> None:
        """A docstring or comment mentioning a verb is not a call.

        This is the shape the pre-fix showcase had: its docstring listed the
        vocabulary while its body drove part of it.
        """
        planted = '"""Exercises action_send_goal and list_actions."""\n# action="echo"\nrun(action="status")\n'
        assert _called_actions(planted) == {"status"}

    def test_a_non_literal_action_is_not_credited(self) -> None:
        """A verb assembled at runtime is not evidence that it was driven."""
        planted = "for verb in VERBS:\n    run(action=verb)\n"
        assert _called_actions(planted) == set()
