"""An ``action`` the documentation names must be one the tool dispatches.

Every ``@tool`` entry point in :mod:`strands_robots.tools` takes a single
``action`` string and dispatches on its value, so that value is an enumerated
vocabulary rather than free text. A page that names an action outside it does
not merely read loosely: the call is refused on the first line a reader runs.

The sibling guard in ``test_docs_python_examples_are_callable`` grades the other
half of the same example - whether the *keywords* a documented call passes are
accepted by the callable - and a wrong action value passes it, because ``action``
is a real parameter of every one of these tools. So
``pose_tool(action="fk", robot_id=..., port=...)`` spends three accepted
keywords and is still refused, and nothing graded the value.

What the refusal looks like varies, and the quieter shapes are the reason this
is worth pinning rather than left to review:

* ``lerobot_camera(action="stream")`` answers ``Unknown action: stream`` and
  names no valid action at all.
* ``serial_tool(action="list")`` does not report an unknown action - it answers
  ``Port parameter required for this action``, so the reader supplies a port and
  gets ``Serial error: could not open port`` next, two steps away from
  ``list_ports``, which needs no port.

The rule is one-directional. The tables name key actions, so naming a subset
is correct and omission is never reported; only an action no tool dispatches
is.

Two table shapes are read. A hand-written row (```| `tool` | ... `"action"` ...```)
and the generated tool reference on ``docs/reference/tools.md``, whose
``{{tools_ref}}`` token ``docs/hooks/tools_ref.py`` expands at build time into
one row per ``@tool`` with an ``actions or parameters`` column
(``<code>action</code>, ...``, or ``params: ...`` for a tool without an
``action``). The generated page is graded as rendered, so a vocabulary the hook
reads differently from the dispatch branches shows up here.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_TOOLS_DIR = _REPO_ROOT / "strands_robots" / "tools"
_TOOLS_PAGE = _REPO_ROOT / "docs" / "reference" / "tools.md"
_TOOLS_HOOK = _REPO_ROOT / "docs" / "hooks" / "tools_ref.py"

# A documented call: ``<tool>(action="<value>"`` anywhere in a page, fenced or
# inline. The action is the first argument at every documented call site.
_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(\s*action\s*=\s*[\"']([^\"']+)[\"']")

# A hand-written tools-table row: ``| `<tool>` | ... `"action"`, `"action"` ... |``.
_ROW = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|(.*?)\|")
_QUOTED = re.compile(r'`"([^"]+)"`')
# A generated tool-reference row: ``| <code>tool</code> | <code>module</code> | does | <code>a</code>, <code>b</code> |``
# or ``... | params: <code>x</code> |`` for a tool that takes no ``action``.
_GENERATED_ROW = re.compile(r"^\|\s*<code>([a-z0-9_]+)</code>\s*\|[^|]*\|[^|]*\|([^|]*)\|\s*$")
_CODE = re.compile(r"<code>([^<]+)</code>")

# A page reflow that hides the tables would otherwise report a clean sweep.
_MINIMUM_CLAIMS = 40
_MINIMUM_TOOLS = 6
_MINIMUM_DISPATCHED = 3


@dataclass(frozen=True)
class _Claim:
    """One action value a page attributes to one tool."""

    page: str
    kind: str
    tool: str
    action: str

    def __str__(self) -> str:
        return f"{self.page} ({self.kind}): {self.tool}(action={self.action!r})"


def _tool_modules() -> dict[str, Path]:
    """Every public ``@tool`` module, keyed by the tool name it exports."""
    return {p.stem: p for p in sorted(_TOOLS_DIR.glob("*.py")) if not p.stem.startswith("_")}


def _string_literals(node: ast.AST) -> set[str]:
    """The string constants ``node`` is, or collects if it is a literal group."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return set()


def _module_string_rosters(tree: ast.Module) -> dict[str, set[str]]:
    """Module-level names bound to a collection of string literals.

    A tool that states its vocabulary once - ``_ACTIONS = ("status", ...)``,
    refused with ``if action not in _ACTIONS`` - names every verb it answers in
    that one constant, so the constant is the roster even though no comparison
    quotes the verbs.
    """
    rosters: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        # frozenset({...}) / set([...]) wrap the literal group they are built from.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id in {"frozenset", "set", "tuple", "list"} and value.args:
                value = value.args[0]
        literals = _string_literals(value)
        if not literals:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                rosters[target.id] = literals
    return rosters


def _dispatched_actions(source: Path) -> set[str]:
    """The action values ``source`` dispatches on.

    Reads both operand positions of a comparison against ``action`` so
    ``"list" == action`` counts, the collection of an ``action in (...)``
    membership test, and the patterns of a ``match action`` statement.

    A comparison against a module-level roster contributes that roster: an
    envelope that refuses everything outside ``_ACTIONS`` and forwards the
    remainder to a delegate module dispatches every name in it, and quotes
    none of them here.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    rosters = _module_string_rosters(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(isinstance(o, ast.Name) and o.id == "action" for o in operands):
                for operand in operands:
                    found |= _string_literals(operand)
                    if isinstance(operand, ast.Name) and operand.id in rosters:
                        found |= rosters[operand.id]
        if isinstance(node, ast.Match) and isinstance(node.subject, ast.Name) and node.subject.id == "action":
            for case in node.cases:
                for pattern in ast.walk(case.pattern):
                    if isinstance(pattern, ast.MatchValue):
                        found |= _string_literals(pattern.value)
    return found


def _graded_pages() -> list[Path]:
    """The operator-facing prose graded here."""
    return sorted(_REPO_ROOT.glob("docs/**/*.md")) + [_REPO_ROOT / "README.md"]


def _rendered(page: Path) -> str:
    """``page`` as the reader sees it: the tool reference has its token expanded by the hook."""
    text = page.read_text(encoding="utf-8")
    if page != _TOOLS_PAGE:
        return text
    spec = importlib.util.spec_from_file_location("docs_hooks_tools_ref", _TOOLS_HOOK)
    assert spec is not None and spec.loader is not None
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    rendered = module.on_page_markdown(text, page=None, config=None, files=None)
    assert rendered != text, "docs/reference/tools.md carries no {{tools_ref}} token for the hook to expand"
    return rendered


def _claims_in(text: str, page: str, tools: dict[str, Path]) -> list[_Claim]:
    """The action values ``text`` attributes to a known tool."""
    claims = [_Claim(page, "call", tool, action) for tool, action in _CALL.findall(text) if tool in tools]
    for line in text.splitlines():
        row = _ROW.match(line)
        if row is not None and row.group(1) in tools:
            claims += [_Claim(page, "table", row.group(1), a) for a in _QUOTED.findall(row.group(2))]
            continue
        generated = _GENERATED_ROW.match(line)
        if generated is None or generated.group(1) not in tools:
            continue
        cell = generated.group(2).strip()
        if cell.startswith("params:"):
            continue  # a tool without an ``action`` lists its parameters instead
        claims += [_Claim(page, "table", generated.group(1), a) for a in _CODE.findall(cell)]
    return claims


def _documented_claims(tools: dict[str, Path]) -> list[_Claim]:
    """Every action value the graded prose attributes to a tool."""
    claims: list[_Claim] = []
    for page in _graded_pages():
        rel = page.relative_to(_REPO_ROOT).as_posix()
        claims += _claims_in(_rendered(page), rel, tools)
    return claims


def _claimed_tools() -> set[str]:
    """The tools some graded page attributes at least one action to."""
    return {c.tool for c in _documented_claims(_tool_modules())}


def _undispatched(claims: list[_Claim], tools: dict[str, Path]) -> list[_Claim]:
    """The claims naming an action the tool does not dispatch."""
    vocabulary = {tool: _dispatched_actions(path) for tool, path in tools.items()}
    return [c for c in claims if c.action not in vocabulary[c.tool]]


class TestEveryDocumentedActionIsDispatched:
    """The action vocabulary the prose names is the one the tools answer."""

    def test_no_page_names_an_action_no_tool_dispatches(self) -> None:
        """A documented action value must reach a dispatch branch."""
        tools = _tool_modules()
        offenders = _undispatched(_documented_claims(tools), tools)
        assert not offenders, "documentation names actions no tool dispatches:\n" + "\n".join(
            f"  {o}  (dispatched: {sorted(_dispatched_actions(tools[o.tool]))})" for o in offenders
        )

    def test_the_sweep_reaches_the_prose(self) -> None:
        """A clean result must mean the prose was read, not that nothing was."""
        tools = _tool_modules()
        claims = _documented_claims(tools)
        assert len(claims) >= _MINIMUM_CLAIMS, f"only {len(claims)} action claims found"
        assert len({c.tool for c in claims}) >= _MINIMUM_TOOLS
        assert {c.kind for c in claims} == {"call", "table"}, "both claim shapes must be reached"

    @pytest.mark.parametrize("tool", sorted(_claimed_tools()))
    def test_every_claimed_tool_dispatches_a_readable_vocabulary(self, tool: str) -> None:
        """A vocabulary that read short would accept a claim it should report.

        Only the tools the prose names are graded: a tool no page claims never
        has its vocabulary consulted, and several ``@tool`` entry points take no
        ``action`` at all. An empty read is already loud rather than silent -
        every claim about such a tool becomes an offender above.
        """
        dispatched = _dispatched_actions(_tool_modules()[tool])
        assert len(dispatched) >= _MINIMUM_DISPATCHED, f"{tool}: read {sorted(dispatched)}"


class TestTheRuleIsOneDirectional:
    """Naming a subset is correct; the column is "Key actions"."""

    def test_a_documented_subset_is_not_reported(self) -> None:
        """A tool may dispatch more actions than any page lists.

        The generated reference lists every action, so the strict subsets live
        in the hand-written pages: a Learn page calls one or two actions of a
        tool that dispatches many. At least one such tool has to exist for the
        rule to be exercised, and none of its calls may be reported.
        """
        tools = _tool_modules()
        by_tool: dict[str, list[_Claim]] = {}
        for claim in _documented_claims(tools):
            if claim.kind == "call":
                by_tool.setdefault(claim.tool, []).append(claim)
        subsets = {
            tool: claims
            for tool, claims in by_tool.items()
            if {c.action for c in claims} < _dispatched_actions(tools[tool])
        }
        assert subsets, "premise: no page documents a strict subset of any tool's actions"
        for claims in subsets.values():
            assert not _undispatched(claims, tools)


class TestTheGraderIsLoadBearing:
    """The sweep reports a planted claim, and only a wrong one."""

    def test_a_planted_undispatched_action_is_reported(self) -> None:
        """A page naming an action outside the vocabulary must fail."""
        tools = _tool_modules()
        planted = _claims_in('result = serial_tool(action="no_such_action")', "planted.md", tools)
        assert [c.action for c in planted] == ["no_such_action"]
        assert _undispatched(planted, tools) == planted

    def test_a_planted_dispatched_action_is_accepted(self) -> None:
        """A page naming a real action must not be reported."""
        tools = _tool_modules()
        planted = _claims_in('result = serial_tool(action="list_ports")', "planted.md", tools)
        assert [c.action for c in planted] == ["list_ports"]
        assert _undispatched(planted, tools) == []

    def test_a_planted_table_row_is_graded_too(self) -> None:
        """The table column is read, not only the fenced examples."""
        tools = _tool_modules()
        row = '| `pose_tool` | `"store_pose"`, `"no_such_action"` | What |'
        planted = _claims_in(row, "planted.md", tools)
        assert {c.kind for c in planted} == {"table"}
        assert [c.action for c in _undispatched(planted, tools)] == ["no_such_action"]

    def test_a_planted_generated_row_is_graded_too(self) -> None:
        """The generated tool reference's ``actions or parameters`` column is read."""
        tools = _tool_modules()
        row = "| <code>pose_tool</code> | <code>tools.pose</code> | Poses. | <code>store_pose</code>, <code>no_such_action</code> |"
        planted = _claims_in(row, "planted.md", tools)
        assert {c.kind for c in planted} == {"table"}
        assert [c.action for c in _undispatched(planted, tools)] == ["no_such_action"]
        params = "| <code>pose_tool</code> | <code>tools.pose</code> | Poses. | params: <code>robot_id</code> |"
        assert _claims_in(params, "planted.md", tools) == [], "a parameter list is not an action vocabulary"

    def test_a_tool_the_package_does_not_ship_is_not_graded(self) -> None:
        """A same-shaped call to something else degrades to "not graded"."""
        tools = _tool_modules()
        assert _claims_in('other_library(action="whatever")', "planted.md", tools) == []
