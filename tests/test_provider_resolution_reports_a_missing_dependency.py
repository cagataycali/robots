"""A provider whose module needs an absent dependency reports the dependency.

Three packages resolve a provider name to a class the same way, and each ends
its ladder with an auto-discovery rung that imports
``strands_robots.<package>.<provider>``. That import has two distinct failure
modes and they need distinct reports:

* the module does not exist - the provider name is unknown, and the caller wants
  the list of names that *are* known;
* the module exists but something IT imports is absent - the name was right and
  the caller wants the missing dependency.

Collapsing the second into the first sends a caller whose spelling was correct
to go and check the spelling. ``import_policy_class`` and
``import_transform_class`` already separate them, each saying so in its own
``Raises:`` block; ``import_trainer_class`` caught ``ImportError`` and fell
through to ``No trainer registered for provider ... Available trainers: [...]``,
so a missing backend was reported as a name that does not exist.

The rule is graded over a DERIVED inventory rather than a list of three, because
the defect is what happens to the *next* resolver. A resolver is discovered
structurally: a function that hands :func:`importlib.import_module` an f-string
which begins ``strands_robots.`` and interpolates one of that function's own
parameters. That signature selects exactly the shipped three and nothing else,
and :class:`TestTheResolverInventoryIsDerived` pins that a fourth is picked up
on arrival - a hardcoded list equal to today's three would pass every case below
while grading nothing new.

Each resolver is then driven for real: a module is planted on the package's
``__path__`` whose body imports a package that is not installed, so the
production path runs against the exact ``ModuleNotFoundError`` the import system
raises. The assertions are deliberately disposition-agnostic - one resolver
re-raises and another wraps the error to add an install hint, and both satisfy
"reports the dependency" - so this file does not freeze either choice.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
import tomllib
from typing import Any

import pytest

import strands_robots

#: A distribution nothing installs, so importing it always fails.
_ABSENT_DEP = "a_backend_sdk_that_is_not_installed_xyz"

#: Package root of the library under test, derived from the package itself.
_PACKAGE_ROOT = pathlib.Path(strands_robots.__file__).resolve().parent


class _Resolver:
    """One discovered provider-module resolver.

    Attributes:
        module: Dotted path of the module defining the resolver.
        function: Name of the resolver function.
        package: Dotted path of the package its auto-discovery rung searches.
    """

    def __init__(self, module: str, function: str, package: str) -> None:
        self.module = module
        self.function = function
        self.package = package

    def __repr__(self) -> str:
        return f"{self.module}.{self.function}"

    def resolve(self, provider: str) -> Any:
        """Call the resolver, importing its module fresh through its dotted path."""
        return getattr(importlib.import_module(self.module), self.function)(provider)


def resolvers_in_source(source: str, module: str) -> list[_Resolver]:
    """Find every provider-module resolver defined in one module's source.

    A resolver hands :func:`importlib.import_module` an f-string that begins
    ``strands_robots.`` and interpolates one of the enclosing function's own
    parameters - that is what makes the imported module name caller-controlled,
    which is what makes "no such module" ambiguous with "no such provider".
    The f-string is accepted either inline at the call or through a local name
    bound to it earlier in the same function.

    Args:
        source: Python source text.
        module: Dotted path to record for anything found.

    Returns:
        One :class:`_Resolver` per discovered function, ordered by appearance.
    """
    found: list[_Resolver] = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        parameters = {a.arg for a in function.args.args} | {a.arg for a in function.args.kwonlyargs}
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "import_module"
                and call.args
            ):
                continue
            for template in _candidate_templates(call.args[0], function):
                literal = "".join(
                    v.value for v in template.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
                interpolated = {ast.unparse(v.value) for v in template.values if isinstance(v, ast.FormattedValue)}
                if literal.startswith("strands_robots.") and (interpolated & parameters):
                    found.append(_Resolver(module, function.name, literal.rstrip(".")))
    return found


def _candidate_templates(argument: ast.expr, function: ast.AST) -> list[ast.JoinedStr]:
    """Return the f-strings ``argument`` can be, inline or via a local name."""
    if isinstance(argument, ast.JoinedStr):
        return [argument]
    if not isinstance(argument, ast.Name):
        return []
    return [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == argument.id
        and isinstance(node.value, ast.JoinedStr)
    ]


def discover_resolvers() -> list[_Resolver]:
    """Walk the shipped package for every provider-module resolver."""
    found: list[_Resolver] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        dotted = ".".join(path.relative_to(_PACKAGE_ROOT.parent).with_suffix("").parts)
        found.extend(resolvers_in_source(path.read_text(encoding="utf-8"), dotted))
    return found


#: A concrete shipped base per package, for planting a usable provider.
_MOCK_BASES = {
    "strands_robots.policies": ("strands_robots.policies.mock", "MockPolicy"),
    "strands_robots.training": ("strands_robots.training.mock", "MockTrainer"),
    "strands_robots.transforms": ("strands_robots.transforms.mock", "MockTransform"),
}

#: Every resolver in the shipped tree, discovered once at collection time.
RESOLVERS = discover_resolvers()


def _ids(resolvers: list[_Resolver]) -> list[str]:
    """Single-token parametrize ids, so ``-k`` can address one resolver."""
    return [r.function for r in resolvers]


@pytest.fixture
def plant(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Any:
    """Return a helper that plants a module on a real package's search path.

    Restores ``__path__`` through ``monkeypatch`` and drops any module the
    planted import managed to register, so nothing survives into the next test.
    """
    planted: list[str] = []

    def _plant(package: str, provider: str, body: str) -> None:
        (tmp_path / f"{provider}.py").write_text(f'"""Planted for one test."""\n{body}', encoding="utf-8")
        module = importlib.import_module(package)
        monkeypatch.setattr(module, "__path__", [*module.__path__, str(tmp_path)])
        planted.append(f"{package}.{provider}")
        importlib.invalidate_caches()

    yield _plant

    for name in planted:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _top_level_imports(path: pathlib.Path) -> set[str]:
    """Return the top-level packages ``path`` imports at module scope."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _dependency_is_named(error: BaseException, dependency: str) -> bool:
    """Whether ``dependency`` is named anywhere in the raised error's chain.

    One resolver re-raises the original ``ModuleNotFoundError`` and another
    wraps it to add an install hint, keeping the original as ``__cause__``.
    Both name the dependency; this reads either without preferring one.
    """
    current: BaseException | None = error
    while current is not None:
        if dependency in str(current):
            return True
        current = current.__cause__
    return False


class TestTheResolverInventoryIsDerived:
    """The rule below is graded over a discovered set, not a list of three."""

    def test_the_scan_finds_every_shipped_provider_resolver(self) -> None:
        """All three ladders are discovered, and nothing else is."""
        assert {r.function for r in RESOLVERS} == {
            "import_policy_class",
            "import_trainer_class",
            "import_transform_class",
        }, f"discovered {RESOLVERS!r}"

    def test_each_resolver_names_the_package_its_fallback_searches(self) -> None:
        """The recorded package is the one the auto-discovery rung imports from."""
        assert {(r.function, r.package) for r in RESOLVERS} == {
            ("import_policy_class", "strands_robots.policies"),
            ("import_trainer_class", "strands_robots.training"),
            ("import_transform_class", "strands_robots.transforms"),
        }

    def test_a_fourth_resolver_is_discovered_on_arrival(self) -> None:
        """A newly added ladder is graded without editing this file.

        The whole value of deriving the inventory: a hardcoded list equal to
        today's three would pass every other case here and pick up nothing.
        """
        source = (
            "import importlib\n"
            "def import_widget_class(provider):\n"
            "    return importlib.import_module(f'strands_robots.widgets.{provider}')\n"
        )
        found = resolvers_in_source(source, "strands_robots.widgets.factory")
        assert [(r.function, r.package) for r in found] == [("import_widget_class", "strands_robots.widgets")]

    def test_an_import_of_a_fixed_module_is_not_a_provider_resolver(self) -> None:
        """Precision: only a caller-controlled module name is ambiguous.

        A resolver that imports a module named by the registry rather than by
        the caller cannot confuse "unknown provider" with "absent dependency",
        so it is deliberately out of scope.
        """
        source = (
            "import importlib\n"
            "def load_declared(config):\n"
            "    return importlib.import_module(config['module'])\n"
            "def load_literal(provider):\n"
            "    return importlib.import_module('strands_robots.training.base')\n"
        )
        assert resolvers_in_source(source, "strands_robots.x") == []


class TestEveryProviderResolverSurfacesAMissingDependency:
    """A provider module that exists but cannot import reports why.

    The headline case is an absent backend distribution. The third cell covers
    the neighbour that is not a distribution problem at all - a module that
    raises ``ImportError`` on purpose - because both are failures of a module
    that IS there, and neither is a provider name that is not.
    """

    @pytest.mark.parametrize("resolver", RESOLVERS, ids=_ids(RESOLVERS))
    def test_the_absent_dependency_is_named_rather_than_the_provider(self, resolver: _Resolver, plant: Any) -> None:
        plant(resolver.package, "heavybackend", f"import {_ABSENT_DEP}  # noqa: F401\n")
        with pytest.raises(ImportError) as excinfo:
            resolver.resolve("heavybackend")
        assert _dependency_is_named(excinfo.value, _ABSENT_DEP), str(excinfo.value)

    @pytest.mark.parametrize("resolver", RESOLVERS, ids=_ids(RESOLVERS))
    def test_the_report_does_not_send_the_caller_to_check_the_name(self, resolver: _Resolver, plant: Any) -> None:
        """The available-providers list belongs to the other failure only.

        Offering it here is the misdirection: the caller's name was correct.
        """
        plant(resolver.package, "listbackend", f"import {_ABSENT_DEP}  # noqa: F401\n")
        with pytest.raises(ImportError) as excinfo:
            resolver.resolve("listbackend")
        assert "vailable" not in str(excinfo.value), str(excinfo.value)

    @pytest.mark.parametrize("resolver", RESOLVERS, ids=_ids(RESOLVERS))
    def test_a_module_that_raises_import_error_itself_is_not_an_unknown_provider(
        self, resolver: _Resolver, plant: Any
    ) -> None:
        """Not every failed import is a missing distribution.

        A partially initialised module raises a plain ``ImportError`` with no
        ``name``, and a vendored SDK that refuses to load raises one on purpose.
        Neither means the provider name was wrong, so neither may be answered
        with the available-providers list.
        """
        plant(resolver.package, "brokenbackend", 'raise ImportError("this backend refuses to load")\n')
        with pytest.raises(ImportError) as excinfo:
            resolver.resolve("brokenbackend")
        assert "refuses to load" in str(excinfo.value) or "refuses to load" in str(excinfo.value.__cause__)


class TestAnUnregisteredProviderIsStillUnregistered:
    """Over-reach controls: the two failures must stay distinguishable.

    Every expectation here also held before the fix - a resolver that reported
    everything as a dependency problem would be just as wrong in the other
    direction.
    """

    @pytest.mark.parametrize("resolver", RESOLVERS, ids=_ids(RESOLVERS))
    def test_a_name_with_no_module_is_refused_with_the_available_set(self, resolver: _Resolver) -> None:
        with pytest.raises(ValueError) as excinfo:
            resolver.resolve("no_such_provider_anywhere_xyz")
        message = str(excinfo.value)
        assert "no_such_provider_anywhere_xyz" in message, message
        assert "vailable" in message, message

    @pytest.mark.parametrize("resolver", RESOLVERS, ids=_ids(RESOLVERS))
    def test_a_module_that_imports_but_declares_no_class_is_still_refused(
        self, resolver: _Resolver, plant: Any
    ) -> None:
        """Importable and empty is an unusable provider, not a broken install."""
        plant(resolver.package, "emptybackend", "")
        with pytest.raises(ValueError) as excinfo:
            resolver.resolve("emptybackend")
        assert "vailable" in str(excinfo.value), str(excinfo.value)

    @pytest.mark.parametrize("resolver", RESOLVERS, ids=_ids(RESOLVERS))
    def test_a_usable_planted_provider_still_resolves(self, resolver: _Resolver, plant: Any) -> None:
        """Narrowing the handler must not stop auto-discovery working at all.

        The base class is reached through its module rather than imported by
        name, so the only subclass in the planted module's namespace is the one
        it defines - the rung scans ``dir()``, which would otherwise also offer
        the base the planted class was built from.
        """
        module, attribute = _MOCK_BASES[resolver.package]
        plant(
            resolver.package,
            "goodbackend",
            f"import {module} as base\n\n\nclass Planted(base.{attribute}):\n    pass\n",
        )
        assert resolver.resolve("goodbackend").__name__ == "Planted"


class TestTheImportSystemNamesTheModuleItCouldNotFind:
    """Premise: the guard's discriminator is a real property of the exception."""

    @pytest.mark.parametrize("resolver", RESOLVERS, ids=_ids(RESOLVERS))
    def test_a_missing_submodule_reports_its_own_full_dotted_name(self, resolver: _Resolver) -> None:
        """``exc.name`` is how "no such provider module" is told from anything else."""
        missing = f"{resolver.package}.no_such_provider_anywhere_xyz"
        with pytest.raises(ModuleNotFoundError) as excinfo:
            importlib.import_module(missing)
        assert excinfo.value.name == missing


class TestTheShippedTreeCanReachThisFailure:
    """A shipped module under one of these packages needs an optional dependency.

    ``strands_robots.training.rl`` says so itself: "Importing this package
    imports ``torch`` (via the env / algo modules), so it is not imported by
    ``strands_robots.training.__init__``". ``torch`` arrives through an extra, so
    on an install without it this module is present and unimportable - which is
    exactly the input the two failure modes have to be told apart for.
    """

    def test_a_shipped_provider_module_needs_a_dependency_an_extra_ships(self) -> None:
        """``training/rl`` is discoverable, imports ``torch``, and says so."""
        package = _PACKAGE_ROOT / "training" / "rl" / "__init__.py"
        assert package.is_file(), f"the auto-discoverable module is gone: {package}"
        assert "imports ``torch``" in package.read_text(encoding="utf-8")
        assert "torch" in _top_level_imports(_PACKAGE_ROOT / "training" / "rl" / "env.py")

    def test_that_dependency_is_declared_as_an_extra_rather_than_required(self) -> None:
        """So an install really can have the module and not the dependency."""
        pyproject = tomllib.loads((_PACKAGE_ROOT.parent / "pyproject.toml").read_text(encoding="utf-8"))
        required = " ".join(pyproject["project"]["dependencies"])
        assert "torch" not in required, f"torch is a hard requirement: {required}"
        optional = pyproject["project"]["optional-dependencies"]
        shipping = sorted(e for e, deps in optional.items() if any(d.startswith("torch") for d in deps))
        assert shipping, f"no extra ships torch: {sorted(optional)}"
