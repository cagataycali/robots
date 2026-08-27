"""A lazily-mapped tool name read off the tools package is not a module.

:mod:`strands_robots.tools` maps each tool name to the ``@tool`` object living
inside the submodule of the *same* name, and caches the result into the package
namespace::

    "pose_tool": (".pose_tool", "pose_tool"),

So ``from strands_robots.tools import pose_tool`` has two possible resolutions,
decided by which import ran first anywhere in the process. CPython's
``_handle_fromlist`` imports the submodule only when the attribute is *absent*;
here the lookup triggers the package ``__getattr__``, which succeeds, so the
submodule is never imported and the name binds to the tool object instead.

The two are not interchangeable. A ``DecoratedFunctionTool`` carries no
``__file__``, no ``__spec__``, and none of the module's private names, so a
source that binds a name this way and then reads it as a module passes or fails
on the import order of the whole process rather than on the behavior it is
about. The failure surfaces as ``AttributeError`` naming the attribute, which
points at the read rather than at the import that decided it.

Both halves of the rule are derived, so a tool added to the mapping and a call
site added to any scanned tree are graded on arrival:

- which names are ambiguous at all - a mapped name that is also a submodule;
- which attributes only a module can answer - asked of the tool object itself,
  rather than listed here, because that list is what drifts.

The sweep over this project's own sources is a regression tripwire: it holds no
grading power of its own once the last ambiguous read is gone, because there is
then nothing for it to classify. The power lives in
:class:`TestTheRuleFlagsTheReadItIsFor`, which runs the same predicate over
constructed sources covering both outcomes.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import strands_robots.tools as tools_package

_TOOLS_ROOT = Path(tools_package.__file__).resolve().parent
_REPO_ROOT = _TOOLS_ROOT.parent.parent

# Every tree whose sources are this project's own. Production code is scanned
# too: the ambiguity is a property of the import, not of the caller, so a
# shipped module that grew the same read would be the same defect.
_SCANNED_ROOTS = ("strands_robots", "tests", "tests_integ", "scripts")


def _shadowable_names() -> set[str]:
    """The mapped names that a submodule of the same name also claims."""
    return {name for name in tools_package._LAZY_IMPORTS if (_TOOLS_ROOT / f"{name}.py").exists()}


def _tool_object(name: str) -> object:
    """Resolve ``name`` to its tool object, whichever import ran first.

    Going through the package's own mapping and asking the submodule for the
    named attribute is the one spelling a module cannot answer:
    ``getattr(tools_package, name)`` returns whichever of the two this process
    happened to bind, which is the ambiguity under test rather than a way to
    measure it.
    """
    rel_module, attr_name = tools_package._LAZY_IMPORTS[name]
    return getattr(importlib.import_module(rel_module, tools_package.__name__), attr_name)


def _module_only_attributes(name: str) -> set[str]:
    """Attributes the submodule carries that its tool object cannot answer."""
    rel_module, _ = tools_package._LAZY_IMPORTS[name]
    module = importlib.import_module(rel_module, tools_package.__name__)
    tool = _tool_object(name)
    return {attr for attr in dir(module) if not hasattr(tool, attr)}


def _module_reads(source: str) -> list[tuple[int, str]]:
    """Reads of a package-bound tool name that only a module could answer.

    Args:
        source: Python source to classify.

    Returns:
        ``(lineno, expression)`` for every attribute read whose target was bound
        by ``from strands_robots.tools import <ambiguous name>`` and which the
        tool object cannot serve. An attribute both resolutions carry -
        ``tool_name``, ``__name__`` - is not reported: those reads mean the same
        thing either way, and refusing them would refuse working code.
    """
    shadowable = _shadowable_names()
    tree = ast.parse(source)

    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == tools_package.__name__ and node.level == 0:
            for alias in node.names:
                if alias.name in shadowable:
                    bound[alias.asname or alias.name] = alias.name

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in bound
            and not hasattr(_tool_object(bound[node.value.id]), node.attr)
        ):
            found.append((node.lineno, ast.unparse(node)))
    return sorted(found)


def _scanned_sources() -> dict[Path, str]:
    """Every Python source under the scanned trees, keyed by path."""
    sources: dict[Path, str] = {}
    for root in _SCANNED_ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            sources[path] = path.read_text(encoding="utf-8")
    return sources


class TestTheTwoResolutionsReallyDiffer:
    """The premise: the name resolves to two objects that are not swappable."""

    def test_the_tool_object_is_not_the_submodule(self) -> None:
        """Otherwise there would be no ambiguity to guard."""
        for name in sorted(_shadowable_names()):
            rel_module, _ = tools_package._LAZY_IMPORTS[name]
            module = importlib.import_module(rel_module, tools_package.__name__)
            assert _tool_object(name) is not module, name

    def test_the_tool_object_cannot_answer_a_module_only_attribute(self) -> None:
        """``__file__`` is the read that broke, so it is named explicitly.

        The whole import machinery a module carries is absent, not just that one
        attribute, which is why the derived set is asserted to hold all of it:
        a helper narrowed to the single known read would stop describing the
        difference the rule is about.
        """
        for name in sorted(_shadowable_names()):
            module_only = _module_only_attributes(name)
            assert {"__file__", "__spec__", "__loader__", "__package__"} <= module_only, (name, sorted(module_only))
            assert not hasattr(_tool_object(name), "__file__"), name

    def test_an_attribute_both_carry_is_not_module_only(self) -> None:
        """The rule is "the tool cannot answer it", not "it is a dunder".

        ``__name__`` is carried by both resolutions, so a read of it means the
        same thing either way and must not be reported.
        """
        for name in sorted(_shadowable_names()):
            tool = _tool_object(name)
            assert hasattr(tool, "__name__"), name
            assert "__name__" not in _module_only_attributes(name), name
        assert hasattr(_tool_object("pose_tool"), "tool_name")


class TestNoSourceReadsALazyToolNameAsAModule:
    """The tripwire, over this project's own sources."""

    def test_no_scanned_source_reads_one_as_a_module(self) -> None:
        offenders = {
            path.relative_to(_REPO_ROOT).as_posix(): reads
            for path, source in _scanned_sources().items()
            if (reads := _module_reads(source))
        }
        assert offenders == {}, (
            "these reads resolve a tool object and then use it as a module, so they "
            f"depend on an unrelated import running first: {offenders}. Import the "
            "submodule directly - `import strands_robots.tools.<name> as ...` or "
            "`from strands_robots.tools.<name> import <name>`."
        )

    def test_the_scan_reaches_the_trees_it_claims(self) -> None:
        """A mistyped root would empty the sweep without failing it."""
        for root in _SCANNED_ROOTS:
            assert (_REPO_ROOT / root).is_dir(), root
        sources = _scanned_sources()
        assert len(sources) > 500, len(sources)
        assert any(path.name == "test_lazy_tool_name_is_not_read_as_a_module.py" for path in sources)


class TestTheRuleFlagsTheReadItIsFor:
    """Where the grading power lives: constructed sources, both outcomes.

    The sweep above classifies whatever the trees happen to contain, so it
    reports nothing once the ambiguous reads are gone. These cases keep the
    predicate itself measured, and pin the two unambiguous import forms as
    accepted so the rule cannot be satisfied by refusing them too.
    """

    _AMBIGUOUS_THEN_MODULE_READ = (
        "from strands_robots.tools import pose_tool as pose_mod\npose_mod.pose_tool(action='list_poses')\n"
    )
    _AMBIGUOUS_THEN_DUNDER_READ = "from strands_robots.tools import use_rosbridge as ur\nroot = ur.__file__\n"
    _AMBIGUOUS_THEN_SHARED_READ = "from strands_robots.tools import pose_tool\nname = pose_tool.tool_name\n"
    _SUBMODULE_IMPORT = "import strands_robots.tools.pose_tool as pose_mod\npose_mod.pose_tool(action='list_poses')\n"
    _SUBMODULE_FROM_IMPORT = "from strands_robots.tools.pose_tool import pose_tool\npose_tool(action='list_poses')\n"

    _CASES = (
        pytest.param(_AMBIGUOUS_THEN_MODULE_READ, True, id="ambiguous-import-then-tool-name-read"),
        pytest.param(_AMBIGUOUS_THEN_DUNDER_READ, True, id="ambiguous-import-then-dunder-read"),
        pytest.param(_AMBIGUOUS_THEN_SHARED_READ, False, id="ambiguous-import-but-shared-attribute"),
        pytest.param(_SUBMODULE_IMPORT, False, id="submodule-import"),
        pytest.param(_SUBMODULE_FROM_IMPORT, False, id="submodule-from-import"),
    )

    @pytest.mark.parametrize(("source", "flagged"), _CASES)
    def test_the_predicate_grades_the_constructed_source(self, source: str, flagged: bool) -> None:
        assert bool(_module_reads(source)) is flagged, source

    def test_the_constructed_cases_reach_both_outcomes(self) -> None:
        """Otherwise the grid could pass by only ever asking one question."""
        outcomes = {case.values[1] for case in self._CASES}
        assert outcomes == {True, False}
