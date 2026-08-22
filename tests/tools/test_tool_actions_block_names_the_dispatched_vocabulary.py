"""An ``Actions:`` block is the vocabulary an agent picks a verb from.

Every ``@tool`` in this package dispatches on a string ``action``, and the
docstring's ``Actions:`` block is where a caller learns which strings that is.
It is not merely documentation: the ``@tool`` decorator publishes the docstring
as ``tool_spec["description"]``, so the block is the literal text handed to a
model, and an entry there is a promise the tool answers that verb.

``pose_tool`` advertised ``"calibrate_motor": Interactive motor calibration``
alongside thirteen real entries, in the same quoted-and-colon format, from its
original commit onwards. ``action == "calibrate_motor"`` has never existed in any
commit, so the tool answered its own documented verb with
``Unknown action: calibrate_motor`` - byte-identical to the refusal for a
misspelling, and followed by an ``Available actions:`` list that omitted it. The
tool contradicted itself in one file, and the half a model reads first was the
wrong one. Nothing compared the two, because the guards on this contract sit at
the *published-schema* boundary of the MuJoCo ``Simulation`` tool
(``tests/simulation/mujoco/test_agent_boundary_refuses_unpublished_action.py``)
and grade an enum, not a prose block.

These tests pin both directions for every ``@tool`` module in
:mod:`strands_robots.tools` whose docstring has an ``Actions:`` block: an entry
naming a verb the module dispatches nowhere fails here, and so does a dispatched
verb the block never names. Nine of the ten surfaces were already exact, so this
grades a convention the package already keeps.

The parser is calibrated against the three delimiter styles that actually ship -
bare (``status - ...``), double-quoted (``- "store_pose": ...``) and
double-backticked (``- ``start``: ...``) - and against the two shapes that are
*not* entries: a wrapped continuation line, and a capitalised sub-heading such as
``Motor Control:`` that groups them. Two separate rules exclude a continuation,
and both are needed. A name must be followed *directly* by ``:`` or ``-``, which
is what keeps ``summary (JSON object: strategy...)`` inside
``harness_memory``'s ``save_trace`` description from reading as an entry - that
one is followed by ``(``. A wrap that does begin ``word:`` is shaped exactly like
an entry, so only its deeper indent separates the two; no block ships that shape
today, but noting an action's own parameters underneath it is the obvious way to
reach it. :class:`TestTheParserIsCalibratedAgainstTheShippedStyles` pins each
calibration, so a parser that stopped recognising one would fail here rather than
silently grading nothing.

The scan walks modules by AST, so it needs none of the optional hardware
dependencies installed, and :class:`TestTheScanIsNonVacuous` fails if a
re-narrowed scan reports a clean sweep over less than the tool package.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import strands_robots.tools as tools_package

# Derived from an imported symbol rather than a path literal, so a moved package
# cannot leave this scanning an empty tree while reporting success.
_TOOLS_ROOT = Path(inspect.getfile(tools_package)).parent

# A block entry: an optional bullet, then the verb, optionally delimited by
# ``"``, ``'``, `````` or ```` `` ````, then ``:`` or ``-`` before its
# description. Lower-case initial letter, which is what separates an entry from
# a capitalised sub-heading like ``Pose Management:``.
_ENTRY = re.compile(r"^(?:[-*]\s+)?(?:``|`|\"|')?([a-z][a-z0-9_]*)(?:``|`|\"|')?\s*(?::|-\s)")

# Enough surfaces that a re-narrowed or mis-rooted scan cannot pass quietly.
_MINIMUM_SURFACES = 8


def named_actions(doc: str) -> frozenset[str] | None:
    """Verbs listed as entries in ``doc``'s ``Actions:`` block.

    Args:
        doc: A dedented docstring, as :func:`ast.get_docstring` returns.

    Returns:
        The verbs named, or ``None`` when there is no ``Actions:`` block - which
        callers must distinguish from an empty block, since a surface with no
        block is out of scope rather than documenting nothing.

        Only lines at the block's *shallowest* entry indent count. A wrapped
        description continues its entry at a deeper indent, so a colon inside
        that prose ("``summary (JSON object: strategy...``") is not read as an
        entry of its own.
    """
    header = re.search(r"^([ \t]*)Actions:[ \t]*$", doc, re.MULTILINE)
    if header is None:
        return None
    header_indent = len(header.group(1))
    body: list[str] = []
    for line in doc[header.end() :].splitlines():
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= header_indent:
            break
        body.append(line)
    candidates = [
        (len(line) - len(line.lstrip()), match)
        for line, match in ((line, _ENTRY.match(line.strip())) for line in body)
        if match is not None
    ]
    if not candidates:
        return frozenset()
    entry_indent = min(indent for indent, _ in candidates)
    return frozenset(match.group(1) for indent, match in candidates if indent == entry_indent)


def dispatched_actions(tree: ast.Module) -> frozenset[str]:
    """Every string a module compares its ``action`` against.

    Args:
        tree: A parsed tool module.

    Returns:
        The literals reached by ``action == "x"``, ``action in ("x", ...)`` and
        ``match action: case "x"``. Compared by attribute suffix, so a local
        ``action`` and a ``payload.action`` are both read.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            subject = ast.unparse(node.left)
            for operator, comparator in zip(node.ops, node.comparators, strict=True):
                if not subject.endswith("action"):
                    continue
                if isinstance(operator, ast.Eq | ast.NotEq):
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                        found.add(comparator.value)
                elif isinstance(operator, ast.In | ast.NotIn) and isinstance(
                    comparator, ast.Tuple | ast.List | ast.Set
                ):
                    for element in comparator.elts:
                        if isinstance(element, ast.Constant) and isinstance(element.value, str):
                            found.add(element.value)
        elif isinstance(node, ast.Match) and ast.unparse(node.subject).endswith("action"):
            for case in node.cases:
                for pattern in ast.walk(case.pattern):
                    if (
                        isinstance(pattern, ast.MatchValue)
                        and isinstance(pattern.value, ast.Constant)
                        and isinstance(pattern.value.value, str)
                    ):
                        found.add(pattern.value.value)
    return frozenset(found)


def _graded_surfaces() -> list[tuple[str, frozenset[str], frozenset[str]]]:
    """Every tool module carrying an ``Actions:`` block, with both vocabularies.

    Returns:
        ``(label, named, dispatched)`` per surface, sorted by label.
    """
    surfaces: list[tuple[str, frozenset[str], frozenset[str]]] = []
    for path in sorted(_TOOLS_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dispatched = dispatched_actions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            named = named_actions(ast.get_docstring(node) or "")
            if named is None:
                continue
            owner = getattr(node, "name", path.stem)
            surfaces.append((f"{path.name}::{owner}", named, dispatched))
    return sorted(surfaces)


class TestEveryNamedActionIsDispatched:
    """An entry in an ``Actions:`` block is a verb the tool answers."""

    def test_no_block_advertises_an_action_the_tool_dispatches_nowhere(self) -> None:
        """A phantom entry is refused exactly like a misspelling.

        ``pose_tool`` named ``calibrate_motor`` here and dispatched thirteen
        verbs, so the one entry a caller could not use was indistinguishable
        from the thirteen it could.
        """
        phantoms = {
            label: sorted(named - dispatched) for label, named, dispatched in _graded_surfaces() if named - dispatched
        }
        assert not phantoms, (
            "an Actions: block advertises a verb the module dispatches nowhere, so the tool "
            f"answers its own documented action with its Unknown-action refusal: {phantoms}"
        )

    def test_no_dispatched_action_is_left_out_of_the_block(self) -> None:
        """A verb absent from the block is undiscoverable but selectable."""
        unnamed = {
            label: sorted(dispatched - named) for label, named, dispatched in _graded_surfaces() if dispatched - named
        }
        assert not unnamed, f"a dispatched action is named in no Actions: block: {unnamed}"


class TestPoseToolDescribesItselfConsistently:
    """``pose_tool`` states its vocabulary twice, and both must be the same."""

    def test_the_refusal_lists_exactly_the_dispatched_actions(self) -> None:
        """The ``Available actions:`` list is the block's other half.

        It was already exact while the block was not, which is how the tool came
        to contradict itself; grading only one half leaves that reachable.
        """
        source = (_TOOLS_ROOT / "pose_tool.py").read_text(encoding="utf-8")
        dispatched = dispatched_actions(ast.parse(source))
        listed = re.search(r'"Available actions: ((?:[^"]|"\s*\n\s*")+)"', source)
        assert listed is not None, "pose_tool no longer lists its actions in the refusal"
        advertised = {name.strip() for name in re.sub(r'"\s*\n\s*"', "", listed.group(1)).split(",")}
        assert advertised == set(dispatched), (
            f"pose_tool's refusal lists {sorted(advertised)} but dispatches {sorted(dispatched)}"
        )

    def test_calibration_is_pointed_at_the_tools_that_own_it(self) -> None:
        """Dropping the entry must not leave calibration unfindable.

        ``lerobot_teleoperate`` sets the convention: a tool that does not own
        calibration names the one that does.
        """
        doc = inspect.getdoc(getattr(tools_package.pose_tool, "__wrapped__", tools_package.pose_tool)) or ""
        assert "lerobot_calibrate" in doc, "pose_tool no longer says where stored calibrations are managed"


class TestTheParserIsCalibratedAgainstTheShippedStyles:
    """The five shapes the block parser has to get right."""

    @pytest.mark.parametrize(
        ("style", "line"),
        [
            ("bare", "    status         - roslibpy availability"),
            ("double-quoted", '        - "store_pose": Store current robot pose'),
            ("double-backticked", "        - ``start``: Launch an inference service"),
        ],
    )
    def test_every_shipped_delimiter_style_is_read_as_an_entry(self, style: str, line: str) -> None:
        """All three appear in shipped blocks, so all three must parse."""
        named = named_actions(f"Summary.\n\nActions:\n{line}\n")
        assert named, f"the {style} style is no longer read as an entry"

    def test_a_wrapped_continuation_line_is_not_an_entry(self) -> None:
        """A continuation is excluded by its deeper indent.

        A wrapped description that itself begins ``word:`` - the natural way to
        note an action's own parameters underneath it - is shaped exactly like an
        entry, so only the indent separates the two.
        """
        doc = 'Summary.\n\nActions:\n        - "start": Launch a service.\n          port: 5555 unless given.\n'
        assert named_actions(doc) == frozenset({"start"})

    def test_a_description_that_merely_contains_a_colon_is_not_an_entry(self) -> None:
        """The terminator rule covers the wraps that actually ship.

        ``harness_memory`` wraps ``save_trace``'s description onto a line
        beginning ``summary (JSON object: ...``: a name has to be followed
        directly by ``:`` or ``-`` to be read as an entry, and this one is
        followed by ``(``.
        """
        doc = (
            "Summary.\n\nActions:\n"
            '        - "save_trace": Store a trace plus\n'
            "        summary (JSON object: strategy, what to avoid).\n"
        )
        assert named_actions(doc) == frozenset({"save_trace"})

    def test_a_capitalised_sub_heading_is_not_an_entry(self) -> None:
        """``pose_tool`` groups its entries under three of these."""
        doc = 'Summary.\n\nActions:\n        Motor Control:\n        - "move_motor": Move one motor\n'
        assert named_actions(doc) == frozenset({"move_motor"})

    def test_a_surface_with_no_block_is_out_of_scope(self) -> None:
        """Demanding a block is a different rule from grading one."""
        assert named_actions("Summary.\n\nArgs:\n    action: what to do\n") is None


class TestThePlantedDefectsAreReported:
    """A clean sweep has to mean the blocks agree, not that nothing is graded."""

    def test_a_planted_phantom_is_reported(self) -> None:
        """The shape this guard exists for."""
        named = named_actions('Summary.\n\nActions:\n        - "real": yes\n        - "phantom": no\n')
        dispatched = dispatched_actions(ast.parse('if action == "real":\n    pass\n'))
        assert named is not None
        assert sorted(named - dispatched) == ["phantom"]

    def test_a_planted_unnamed_dispatch_is_reported(self) -> None:
        """The converse direction, over the same parser."""
        named = named_actions('Summary.\n\nActions:\n        - "real": yes\n')
        dispatched = dispatched_actions(
            ast.parse('if action == "real":\n    pass\nelif action == "undocumented":\n    pass\n')
        )
        assert named is not None
        assert sorted(dispatched - named) == ["undocumented"]

    @pytest.mark.parametrize(
        "source",
        [
            'if action in ("a", "b"):\n    pass\n',
            'match action:\n    case "a":\n        pass\n    case "b":\n        pass\n',
        ],
        ids=["membership", "match"],
    )
    def test_every_dispatch_shape_is_collected(self, source: str) -> None:
        """A tool dispatching by membership or ``match`` is graded too."""
        assert dispatched_actions(ast.parse(source)) == frozenset({"a", "b"})


class TestTheScanIsNonVacuous:
    """A mis-rooted scan must fail rather than report a clean sweep."""

    def test_the_scan_reaches_the_tool_package(self) -> None:
        surfaces = _graded_surfaces()
        assert len(surfaces) >= _MINIMUM_SURFACES, (
            f"only {len(surfaces)} Actions: blocks were graded, so the scan is no longer reaching the tool package"
        )

    def test_pose_tool_is_among_the_graded_surfaces(self) -> None:
        """The surface the guard was written for stays in scope."""
        labels = [label for label, _, _ in _graded_surfaces()]
        assert any(label.startswith("pose_tool.py::") for label in labels), labels

    def test_every_graded_surface_names_at_least_one_action(self) -> None:
        """An empty block would satisfy both directions vacuously."""
        empty = [label for label, named, _ in _graded_surfaces() if not named]
        assert not empty, f"an Actions: block names nothing: {empty}"
