"""The documented ABC contract names every public member of ``Policy``.

``docs/learn/policies/index.md`` opens with "The contract", an abridged
``Policy`` fence that is the surface a subclass author reads before writing a
provider, and ``docs/reference/api/policies.md`` renders the whole class through
mkdocstrings. A member absent from both is not merely undocumented - it is
invisible on the pages whose job is to enumerate the contract, so an author
cannot know to override it. That had already happened to eight of the fifteen
public members on the previous hand-written table, ``preflight`` among them,
which is the member deciding whether the simulation renders the whole scene
before a rollout.

The expectation is DERIVED from the class rather than pinned as a literal list,
so a member added to ``Policy`` joins the requirement with no edit here. It is a
biconditional: every public member appears in the fence, or reaches the reader
through the reference page (which, under the site's ``show_if_no_docstring:
false``, renders exactly the members that carry a docstring, and only if the
learn page points at it); and every name in the fence is a public member (a
line for something the class no longer has is equally misleading).
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

from strands_robots.policies.base import Policy

_REPO = pathlib.Path(inspect.getfile(Policy)).parents[2]
_DOC = _REPO / "docs" / "learn" / "policies" / "index.md"
_REFERENCE = _REPO / "docs" / "reference" / "api" / "policies.md"
_REFERENCE_LINK = "reference/api/policies.md"
_MKDOCS = _REPO / "mkdocs.yml"
_HEADING = "## The contract"
_FENCE = re.compile(r"```python[^\n]*\n(.*?)```", re.S)


def _public_members() -> set[str]:
    """Public members declared on ``Policy`` itself (not inherited from object)."""
    return {name for name in vars(Policy) if not name.startswith("_")}


def _contract_section() -> str:
    text = _DOC.read_text(encoding="utf-8")
    start = text.index(_HEADING)
    end = text.index("\n## ", start + len(_HEADING))
    return text[start:end]


def _contract_fence() -> str:
    """The abridged ``class Policy`` fence under "The contract"."""
    fences = [f for f in _FENCE.findall(_contract_section()) if "class Policy" in f]
    assert fences, f"{_DOC.name} has no `class Policy` fence under {_HEADING!r}"
    return fences[0]


def _fence_members() -> set[str]:
    """Every ``def`` and class-level attribute the fence shows."""
    fence = _contract_fence()
    names = set(re.findall(r"^\s*(?:async\s+)?def\s+(\w+)", fence, re.M))
    names |= set(re.findall(r"^\s{4}(\w+)\s*:", fence, re.M))
    return names - {"__init__"}


def _fence_abstract() -> set[str]:
    # One line per decorator, matched without overlapping whitespace classes (py/redos).
    return set(
        re.findall(
            r"@abstractmethod[ \t]*\n(?:[ \t]*@\w+[^\n]*\n)*[ \t]*(?:async[ \t]+)?def[ \t]+(\w+)", _contract_fence()
        )
    )


def _show_if_no_docstring() -> bool:
    """The mkdocstrings option that decides whether an undocumented member renders.

    Read as text, the way the repo's other CI-config pins read mkdocs.yml and
    the workflows: pyyaml is an optional dependency here.
    """
    text = _MKDOCS.read_text(encoding="utf-8")
    assert "mkdocstrings" in text, "mkdocs.yml has no mkdocstrings plugin block"
    match = re.search(r"^\s+show_if_no_docstring:\s*(\w+)\s*$", text, re.M)
    return match is not None and match.group(1).lower() == "true"


def _members_with_docstrings() -> set[str]:
    """Public ``Policy`` members griffe sees a docstring on, so mkdocstrings renders them.

    A def's docstring is its first statement; a class-level attribute's is the
    string literal on the line after it. ``#:`` comments are not docstrings.
    """
    tree = ast.parse(pathlib.Path(inspect.getfile(Policy)).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Policy")
    out: set[str] = set()
    body = cls.body
    for index, node in enumerate(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
            out.add(node.name)
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target is not None and index + 1 < len(body):
            follower = body[index + 1]
            if (
                isinstance(follower, ast.Expr)
                and isinstance(follower.value, ast.Constant)
                and isinstance(follower.value.value, str)
            ):
                out.add(target)
    return {name for name in out if not name.startswith("_")}


def _reference_renders_policy() -> bool:
    """The API page carries a mkdocstrings directive whose members include ``Policy``."""
    text = _REFERENCE.read_text(encoding="utf-8")
    block = re.search(r"^::: strands_robots\.policies\.base\n((?:[ \t]+.*\n?)*)", text, re.M)
    return block is not None and re.search(r"^\s+-\s+Policy\s*$", block.group(1), re.M) is not None


def _learn_page_links_the_reference() -> bool:
    return _REFERENCE_LINK in _contract_section()


class TestTheTableAndTheClassAgree:
    """One fence, one reference page, one class, and no member visible in none of them."""

    def test_the_scan_finds_both_populations(self):
        """Non-vacuity control: an empty side would make either assertion below
        pass for the wrong reason. Floors only - the counts themselves are not
        the claim, so this holds whatever the fence currently lists."""
        assert len(_public_members()) >= 15, "the class scan found no members"
        assert len(_fence_members()) >= 5, "the fence scan found almost nothing"
        assert _reference_renders_policy(), f"{_REFERENCE.name} no longer renders Policy through mkdocstrings"

    def test_every_public_member_is_documented(self):
        """A member neither page shows cannot be overridden by an author who only
        reads the docs. The fence is abridged on purpose; the members it leaves
        out must reach the reader through the reference page, which means they
        carry a docstring and the contract section points at that page."""
        shown = _fence_members()
        rendered = _public_members() if _show_if_no_docstring() else _members_with_docstrings()
        reachable = shown | (rendered if _learn_page_links_the_reference() else set())
        missing = sorted(_public_members() - reachable)
        assert not missing, (
            f"public Policy members absent from the contract fence on {_DOC.name} and not reachable through "
            f"{_REFERENCE_LINK}: {missing}. Either the contract section does not link the reference "
            f"(links: {_learn_page_links_the_reference()}) or the member has no docstring, which "
            "show_if_no_docstring: false hides."
        )

    def test_every_documented_row_is_a_public_member(self):
        """The other direction: a line for a member the class does not have sends
        an author to write something nothing calls."""
        stale = sorted(_fence_members() - _public_members())
        assert not stale, f"contract fence lines naming no public Policy member: {stale}"

    def test_the_abstract_column_matches_the_class(self):
        """``@abstractmethod`` in the fence must mean ``abstractmethod`` on the class,
        so the three members an implementation MUST supply stay the three that
        raise if it does not."""
        documented_abstract = _fence_abstract()
        actual_abstract = set()
        for name, obj in vars(Policy).items():
            if name.startswith("_"):
                continue
            target = getattr(obj, "fget", None) or getattr(obj, "__func__", None) or obj
            if getattr(target, "__isabstractmethod__", False):
                actual_abstract.add(name)
        assert documented_abstract == actual_abstract, (
            f"fence marks abstract={sorted(documented_abstract)}, class says {sorted(actual_abstract)}"
        )

    def test_the_docstring_reader_sees_what_griffe_sees(self):
        """Control: functions with docstrings count, attributes without a following string do not."""
        seen = _members_with_docstrings()
        assert "get_actions" in seen
        assert "control_frequency" not in seen, "control_frequency has no attribute docstring; the reader invented one"
