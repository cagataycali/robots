"""Repo hygiene: the orientation pages' counts are the code's counts.

``docs/concepts/architecture.md`` is the map a contributor reads to size each
subsystem, and the ``## The contract`` fence on ``docs/learn/policies/index.md``
is read as the contract a new implementation conforms to. The old site restated
both in prose and drifted: 4 providers for a package shipping 14, 8 ``@tool``
helpers for 20, four ``Policy`` implementations of fifteen, two of the
contract's three declaration seams, 67 Simulation actions for an enum of 77.

The new site types no count by hand. Every number is a ``{{n:key}}`` token that
``docs/hooks/facts.py`` derives from the tree at build time, the module table is
``{{module_map}}`` from ``scripts/check_import_layers.py``'s ``LAYERS``, and the
provider matrix is ``{{providers:table}}`` from ``registry/policies.json``. That
moves the drift one step back: a hook can derive the wrong number for the word
next to the token, a map can miss a module, and a hand-written contract fence
can still omit a member. So this module grades the hooks against the code and
the pages against the hooks:

* The ``policy_providers`` fact must be the providers ``registry/policies.json``
  ships, the ``tools`` fact the ``@tool`` helpers an AST walk finds, and the
  ``native_drivers`` fact the ``_SHIPPED_DRIVERS`` entries.
* Any page that still qualifies "actions" with a literal is held to the
  published ``tool_spec.json`` enum.
* The contract fence is graded against ``Policy.__abstractmethods__`` (what an
  implementation must supply) and against the *declaration seams*, the "policy
  declares, runtime supplies" family, itself derived from the base class's own
  docstrings rather than listed here.

The seam half is the one with teeth. A policy conforming to the members the
fence names receives no body pose at all: the runtime supplies one only for a
body ``required_bodies`` names, so a whole-body tracker that skips it reads
``base_quat``, the pelvis, which diverges from ``torso_link`` by tens of degrees
once the waist turns.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from strands_robots.policies.base import Policy
from tests._docs_hooks import docs_hook

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "strands_robots"
DOCS = REPO_ROOT / "docs"
ARCHITECTURE = DOCS / "concepts" / "architecture.md"
POLICIES = DOCS / "learn" / "policies" / "index.md"
README = REPO_ROOT / "README.md"

#: The phrase the base class uses to mark a member of the declaration family.
_SEAM_PHRASE = "policy declares, runtime supplies"

#: Words a count qualifies on the architecture page; each must be a token, not a literal.
_COUNTED_NOUNS = ("robots", "categories", "aliases", "providers", "drivers", "tools", "backends", "actions")


def _architecture() -> str:
    return ARCHITECTURE.read_text(encoding="utf-8")


def _contract_fence() -> str:
    """The ``## The contract`` fence of the policies page, the prose contract an implementer reads."""
    text = POLICIES.read_text(encoding="utf-8")
    start = text.find("\n## The contract")
    assert start != -1, "docs/learn/policies/index.md has no '## The contract' section"
    section = text[start:]
    end = section.find("\n## ", 1)
    section = section if end == -1 else section[:end]
    fences = re.findall(r"```python[^\n]*\n(.*?)```", section, re.S)
    assert fences, "the contract section has no python fence to grade"
    return fences[0]


def _tool_count() -> int:
    """``@tool``-decorated functions anywhere under ``strands_robots/``."""
    total = 0
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if any(ast.unparse(d).split("(")[0].strip() == "tool" for d in node.decorator_list):
                total += 1
    return total


def _shipped_providers() -> list[str]:
    """Providers ``registry/policies.json`` ships.

    The shipped registry rather than ``list_providers()``: the pages describe
    what the package contains, and ``list_providers()`` also reports providers a
    caller registered at runtime, which in a test session includes every
    throwaway ``register_policy`` a sibling module left behind.
    """
    registry = json.loads((PACKAGE / "registry" / "policies.json").read_text(encoding="utf-8"))
    return sorted(registry["providers"])


def _shipped_drivers() -> int:
    """Entries of ``_SHIPPED_DRIVERS``, read with ``ast`` like the hooks do."""
    tree = ast.parse((PACKAGE / "drivers" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        target: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign):
            target = node.targets[0]
        if (
            isinstance(node, (ast.AnnAssign, ast.Assign))
            and isinstance(target, ast.Name)
            and target.id == "_SHIPPED_DRIVERS"
            and node.value is not None
        ):
            return len(ast.literal_eval(node.value))
    raise AssertionError("_SHIPPED_DRIVERS not found in strands_robots/drivers/__init__.py")


def _published_action_count() -> int:
    """Actions the MuJoCo tool schema publishes to a model."""
    spec = json.loads((PACKAGE / "simulation" / "mujoco" / "tool_spec.json").read_text(encoding="utf-8"))
    return len(spec["properties"]["action"]["enum"])


def _policy_implementations() -> frozenset[str]:
    """Concrete ``Policy`` implementations, resolved transitively without importing.

    An AST walk rather than ``issubclass`` so an optional dependency missing
    locally cannot silently shrink the set and let a stale number pass.
    """
    bases: dict[str, set[str]] = {}
    for path in sorted((PACKAGE / "policies").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                bases[node.name] = {ast.unparse(b).split(".")[-1] for b in node.bases}
    reached = {"Policy"}
    grew = True
    while grew:
        grew = False
        for name, parents in bases.items():
            if name not in reached and parents & reached:
                reached.add(name)
                grew = True
    return frozenset(reached - {"Policy"})


def _member_doc(name: str) -> str:
    attr = getattr(Policy, name)
    target = attr.fget if isinstance(attr, property) else attr
    return inspect.getdoc(target) or ""


def _declaration_seams() -> frozenset[str]:
    """The "policy declares, runtime supplies" members, derived from the base class.

    A member is in the family when its own docstring names the contract, or when
    a member that does cites it with an ``:attr:`` role, which is how the base
    class records that ``requires_images`` is the precedent the later two follow.
    Deriving it means a fourth seam is graded the day it is added.
    """
    public = {name for name in dir(Policy) if not name.startswith("_")}
    marked: set[str] = set()
    cited: set[str] = set()
    for name in sorted(public):
        doc = _member_doc(name)
        if _SEAM_PHRASE.split(" supplies")[0] in doc:
            marked.add(name)
            cited |= set(re.findall(r":attr:`(?:~[\w.]*\.)?(\w+)`", doc))
    return frozenset(marked | (cited & public))


def _names(text: str, member: str) -> bool:
    """Whether ``text`` names ``member`` as a whole token.

    A dotted path names the thing to import, which is a different fact from the
    member a page is telling an implementer about, so the boundary matters.
    """
    return re.search(rf"(?<![\w.]){re.escape(member)}(?![\w.])", text) is not None


def _docs_pages() -> list[Path]:
    """Every hand-written page plus README; generated robot pages are not prose."""
    pages = [p for p in sorted(DOCS.rglob("*.md")) if "robots" not in p.relative_to(DOCS).parts]
    return [*pages, README]


# --- the architecture page types no count -------------------------------------


@pytest.mark.parametrize("noun", _COUNTED_NOUNS)
def test_the_architecture_page_types_no_count_by_hand(noun: str) -> None:
    """A number next to a counted noun is a ``{{n:key}}`` token, never a literal."""
    literals = re.findall(rf"\b(\d+)\+?\s+(?:\*\*)?{noun}\b", _architecture())
    assert not literals, (
        f"docs/concepts/architecture.md types {literals} {noun} by hand; the page promises its "
        "numbers are generated from the tree, so write {{n:<key>}} instead"
    )


def test_the_architecture_page_uses_the_facts_hook() -> None:
    """Non-vacuity: the page really carries tokens for the guard above to protect."""
    tokens = set(re.findall(r"\{\{\s*n:([a-z_]+)\s*\}\}", _architecture()))
    assert {"policy_providers", "native_drivers", "robots"} <= tokens, (
        f"docs/concepts/architecture.md carries {sorted(tokens)}; the provider, driver and robot counts must be tokens"
    )
    assert "{{module_map}}" in _architecture(), "the module table is the generated {{module_map}}"


# --- the facts hook derives the code's numbers --------------------------------


def test_the_providers_fact_is_the_number_of_providers_that_ship() -> None:
    """``{{n:policy_providers}}`` is read as "providers", so it is ``policies.json``'s count."""
    providers = _shipped_providers()
    stated = docs_hook("facts").numbers()["policy_providers"]
    assert stated == len(providers), (
        f"docs/hooks/facts.py derives policy_providers = {stated}, but registry/policies.json ships "
        f"{len(providers)} providers: {providers}. The pages print the token next to the word "
        "'providers' (architecture.md, index.md), so the hook must count the registry, not the package directories."
    )


def test_the_tools_fact_is_the_number_of_tool_helpers_that_ship() -> None:
    """``{{n:tools}}`` is the ``@tool``-decorated functions that exist."""
    expected = _tool_count()
    stated = docs_hook("facts").numbers()["tools"]
    assert stated == expected, (
        f"docs/hooks/facts.py derives tools = {stated}; an AST walk of strands_robots/ finds {expected}"
    )


def test_the_drivers_fact_is_the_number_of_shipped_drivers() -> None:
    """``{{n:native_drivers}}`` is the ``_SHIPPED_DRIVERS`` roster length."""
    expected = _shipped_drivers()
    stated = docs_hook("facts").numbers()["native_drivers"]
    assert stated == expected, (
        f"docs/hooks/facts.py derives native_drivers = {stated}; _SHIPPED_DRIVERS has {expected} entries"
    )


def test_the_module_map_covers_every_top_level_member() -> None:
    """The generated module table names every module and package, and nothing that is not there."""
    rows = docs_hook("module_map").rows()
    mapped = {member for _, members, _ in rows for member, _ in members}
    real = {
        p.stem if p.is_file() else p.name
        for p in PACKAGE.iterdir()
        if (p.suffix == ".py" or p.is_dir()) and not p.name.startswith("__")
    }
    assert real <= mapped, f"the module map omits {sorted(real - mapped)}"
    phantom = {m for m in mapped - real if not m.startswith("__")}
    assert not phantom, f"the module map lists {sorted(phantom)}, which are not in the package"


# --- action counts, wherever a page states one --------------------------------


@pytest.mark.parametrize("path", _docs_pages(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_a_documented_simulation_action_count_is_the_published_enum(path: Path) -> None:
    """Every action count any page states is the count a model is offered.

    Graded wherever a number qualifies "actions" rather than at fixed pages, so a
    fresh claim added anywhere in the site is held to the same enum.
    """
    expected = _published_action_count()
    text = path.read_text(encoding="utf-8")
    stated = {int(n) for n in re.findall(r"\b(\d+)\s+(?:\*\*)?(?:Simulation |sim )?actions\b", text)}
    wrong = sorted(n for n in stated if n != expected)
    assert not wrong, (
        f"{path.relative_to(REPO_ROOT)} states {wrong} Simulation actions; tool_spec.json publishes {expected}"
    )


def test_some_page_states_the_action_count() -> None:
    """Non-vacuity: the site still tells the reader how many actions the sim tool has."""
    hits = [
        p
        for p in _docs_pages()
        if re.search(r"\b\d+\s+(?:\*\*)?(?:Simulation |sim )?actions\b", p.read_text(encoding="utf-8"))
    ]
    assert hits, "no page states the Simulation action count; the enum guard above grades nothing"


# --- the contract fence -------------------------------------------------------


def test_the_contract_fence_names_every_abstract_member() -> None:
    """A member an implementation *must* supply is a member the fence shows."""
    fence = _contract_fence()
    missing = sorted(m for m in Policy.__abstractmethods__ if not _names(fence, m))
    assert not missing, (
        f"the contract fence on docs/learn/policies/index.md does not show {missing}, which "
        "Policy declares abstract, so an implementer reading it learns an incomplete must-implement set"
    )


def test_the_contract_fence_marks_abstract_exactly_the_abstract_members() -> None:
    """``@abstractmethod`` in the fence sits on the real abstract members and on nothing else."""
    fence = _contract_fence()
    marked = set(re.findall(r"@abstractmethod\s*\n\s*(?:async\s+)?def\s+(\w+)", fence))
    assert marked, "the fence shows no @abstractmethod; a contract with no must-implement set teaches nothing"
    real = set(Policy.__abstractmethods__)
    assert marked == real, (
        f"the fence marks {sorted(marked)} abstract; Policy.__abstractmethods__ is {sorted(real)}: "
        f"wrongly marked {sorted(marked - real)}, unmarked {sorted(real - marked)}"
    )


def test_the_contract_fence_shows_only_real_members() -> None:
    """A ``def`` or attribute in the fence is a member ``Policy`` really has."""
    fence = _contract_fence()
    shown = set(re.findall(r"^\s*(?:async\s+)?def\s+(\w+)", fence, re.M))
    shown |= set(re.findall(r"^\s{4}(\w+)\s*:", fence, re.M))
    shown -= {"__init__"}
    stale = sorted(name for name in shown if not hasattr(Policy, name))
    assert not stale, f"the contract fence shows {stale}, which Policy does not define"


def test_the_contract_fence_names_every_declaration_seam() -> None:
    """A seam the runtime reads is a seam the contract fence names.

    ``requires_images`` alone was named. ``required_bodies`` is the only route to
    a body pose, and ``children`` the only route into a wrapped policy, so a
    fence naming one of the three teaches a contract whose two silent halves are
    the ones a policy cannot recover from by trying harder.
    """
    seams = _declaration_seams()
    fence = _contract_fence()
    missing = sorted(s for s in seams if not _names(fence, s))
    assert not missing, (
        f"the contract fence on docs/learn/policies/index.md does not name {missing}; the base class "
        f"marks {sorted(seams)} as the 'policy declares, runtime supplies' family, and a policy that "
        "skips one is not told it exists"
    )


def test_the_policies_page_states_the_number_of_implementations_that_ship() -> None:
    """An implementation count on the page is the number of implementations."""
    expected = len(_policy_implementations())
    text = POLICIES.read_text(encoding="utf-8")
    stated = {int(n) for n in re.findall(r"\b(\d+)\s+implementations\b", text)}
    wrong = sorted(n for n in stated if n != expected)
    assert not wrong, (
        f"docs/learn/policies/index.md states {wrong} implementations; strands_robots/policies/ defines {expected}"
    )


def test_the_provider_table_is_the_whole_registry() -> None:
    """Naming providers is fine; naming some of them as *the* set is not.

    The old page enumerated four implementations of fifteen as a closed list.
    The new page's matrix is generated, so the guard is on the generator: one
    row per provider ``policies.json`` ships, none missing, none invented.
    """
    table = docs_hook("providers").table()
    rows = [line for line in table.splitlines() if line.startswith("| ") and not line.startswith("| provider")]
    named = {re.sub(r"[\[\]`]", "", cell.split("]")[0]).strip() for cell in (row.split("|")[1] for row in rows)}
    providers = set(_shipped_providers())
    assert named == providers, (
        f"{{{{providers:table}}}} names {sorted(named)}; registry/policies.json ships {sorted(providers)}: "
        f"missing {sorted(providers - named)}, invented {sorted(named - providers)}"
    )


def test_the_seam_family_is_derived_from_the_base_class() -> None:
    """Non-vacuity: the derived family is real members, and it spans all three.

    Without this a refactor that stopped matching the base class's phrasing would
    make the seam guard grade an empty set and report a clean page.
    """
    seams = _declaration_seams()
    assert seams == {"requires_images", "required_bodies", "children"}, (
        f"derived seam family is {sorted(seams)}; expected the three the base class "
        f"marks. Update this pin deliberately if a seam is added or removed."
    )
    for name in seams:
        assert hasattr(Policy, name), f"{name} is not a Policy member"


@pytest.mark.parametrize(
    ("fence", "expected_missing"),
    [
        ("def get_actions(self): ...\ndef set_robot_state_keys(self): ...\ndef provider_name(self): ...", "children"),
        ("def requires_images(self): ...\ndef required_bodies(self): ...\ndef children(self): ...", "get_actions"),
    ],
    ids=["seam-omitted", "abstract-omitted"],
)
def test_the_graders_report_a_planted_omission(fence: str, expected_missing: str) -> None:
    """The rules fire on a fence that omits a member, not merely on the real page."""
    seams = _declaration_seams()
    wanted = set(Policy.__abstractmethods__) | set(seams)
    missing = sorted(m for m in wanted if not _names(fence, m))
    assert expected_missing in missing, f"planted omission of {expected_missing} not reported: {missing}"
