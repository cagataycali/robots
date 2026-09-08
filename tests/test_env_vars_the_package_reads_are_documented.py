"""Every ``STRANDS_*`` environment variable the package reads is documented.

The README's Configuration section calls itself the single source of truth for
these variables, and ``AGENTS.md`` asks that a new one be added there in the
same pull request that introduces it. Nothing graded that: the reference pages
were checked one variable set at a time, each by a test written for the change
that added it (``tests/mesh/test_docs_*_env_var_reference.py``), so a variable
introduced without such a test was undocumented by default and the omission
was silent in the reassuring direction - the code honoured it, the tests that
set it passed, and no page a reader could reach named it.

Measured on the tree this arrived in, the package read 88 distinct ``STRANDS_*``
names and fifteen appeared in no page at all:

- ``STRANDS_MESH_CAMERA_S3_BUCKET`` and ``_PREFIX`` - the two that turn the
  camera S3 offload on. The TTL that only matters once it is on,
  ``STRANDS_MESH_CAMERA_PRESIGN_TTL``, was documented beside where these were
  not, so the README described a knob on a feature it gave no way to enable.
- ``STRANDS_GR00T_REPO_URL`` and ``_TAG`` - the clone source ``build_image``
  fails closed on. Its allowlist, ``STRANDS_GR00T_REPO_URL_ALLOW``, was
  documented in ``docs/security.md`` with no mention of the variable it
  constrains.
- ``STRANDS_MESH_BRIDGE_DEDUP_STRICT``, ``STRANDS_MESH_FILTER_INTERFACES``,
  ``STRANDS_ROBOTS_VERBOSE_MUJOCO`` - each the only spelling of its posture.
- Eight read through a resolver rather than ``os.getenv``: six mesh transport
  bounds (``STRANDS_MESH_MAX_SESSIONS``, ``_MAX_CMD_BYTES``,
  ``_MAX_CAMERA_BYTES``, ``_MAX_SAFETY_BYTES``, ``_CMD_RATE_HZ``,
  ``_SAFETY_RATE_HZ``), the camera privacy switch
  ``STRANDS_MESH_CAMERA_DISABLED`` - read through an import alias - and
  Isaac's ``STRANDS_ISAAC_CAMERA_WARMUP_STEPS``. These are the ones a walk
  that only recognises the direct spellings cannot see.

The population is derived from the package by AST rather than listed here, so
a variable added later is graded on arrival. A read is either a direct one -
``os.getenv``, ``os.environ.get`` / ``setdefault`` / ``[...]`` - or a call to a
function that reads the environment through one of its own parameters
(``_int_env("STRANDS_MESH_MAX_SESSIONS", ...)``); that set of resolvers is
derived from the tree too, to a fixed point so a resolver that delegates to
another is included, and an import alias (``_bool_env as _zc_bool_env``) is
followed. Twenty-nine of the 88 names reach the environment only that way,
so recognising the four direct spellings alone reports a clean tree that is
not one. A page is any of ``README.md`` and
``docs/**/*.md``: ``docs/security.md`` already owns the AWS IoT credentials
and the mesh TLS material, graded by their own reference tests, and this test
does not move them. It also honours the README's shorthand for a family of
sibling names (```STRANDS_MESH_POSE_HZ`, `_IMU_HZ`, ...``) - a suffix counts
only when a documented full name shares its prefix, so a bare suffix with no
sibling documents nothing.

Out of scope, and why: a name that appears only inside a string literal is not
read by this process. ``mesh.iot.bootstrap`` ships the e-stop fan-out Lambda's
source as text and sets that Lambda's ``STRANDS_SAFETY_TABLE`` itself, so the
variable is provisioned rather than exposed, and the AST walk does not see it
by construction.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import strands_robots

PACKAGE = Path(strands_robots.__file__).parent
REPO_ROOT = PACKAGE.parent
PAGES = (REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md")))

#: The prefix every variable this package owns is spelled with. Names read
#: from another tool's namespace (``MUJOCO_GL``, ``ZENOH_CONNECT``,
#: ``GROOT_API_TOKEN``) are that tool's to document and are not graded here.
OWN_PREFIX = "STRANDS_"

#: A whole ``STRANDS_*`` token on a page - not a prefix of a longer name, so a
#: page naming ``STRANDS_GR00T_REPO_URL_ALLOW`` has not named
#: ``STRANDS_GR00T_REPO_URL``.
_FULL_NAME = re.compile(r"(?<![A-Z0-9_])(STRANDS_[A-Z0-9_]+)(?![A-Z0-9_])")

#: The README's sibling shorthand: a backticked ``_SUFFIX`` standing beside a
#: full name it shares a prefix with.
_SHORTHAND = re.compile(r"`(_[A-Z0-9_]+)`")

#: Floors so a walk that silently reads nothing fails rather than passing. The
#: tree this arrived in read 88 names across 107 sites and documented them on
#: 6 pages; both floors sit well below that.
MINIMUM_NAMES_READ = 60
MINIMUM_PAGES_NAMING_ONE = 3


def _environment_key(node: ast.AST) -> str | None:
    """The literal name a read of the environment names, or None.

    Four spellings are reads: ``os.getenv(NAME[, default])``,
    ``os.environ.get(NAME[, default])``, ``os.environ.setdefault(NAME, default)``
    and ``os.environ[NAME]``. The receiver may be ``os.environ`` or a bare
    ``environ`` / ``getenv`` imported by name; a variable key is not graded
    because it names nothing a page could spell.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if not node.args:
            return None
        if isinstance(func, ast.Attribute):
            if func.attr == "getenv":
                key = node.args[0]
            elif func.attr in ("get", "setdefault") and _is_environ(func.value):
                key = node.args[0]
            else:
                return None
        elif isinstance(func, ast.Name) and func.id == "getenv":
            key = node.args[0]
        else:
            return None
    elif isinstance(node, ast.Subscript) and _is_environ(node.value):
        key = node.slice
    else:
        return None
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _direct_key(node: ast.AST) -> ast.AST | None:
    """The key expression of a direct read of the environment, or None."""
    if isinstance(node, ast.Call):
        if not node.args:
            return None
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "getenv" or (func.attr in ("get", "setdefault") and _is_environ(func.value)):
                return node.args[0]
            return None
        if isinstance(func, ast.Name) and func.id == "getenv":
            return node.args[0]
        return None
    if isinstance(node, ast.Subscript) and _is_environ(node.value):
        return node.slice
    return None


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else None


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    """``{local name: imported name}`` for every ``from m import f as g``."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name.rsplit(".", 1)[-1]
    return aliases


def _read_key(node: ast.AST, resolvers: dict[str, int], aliases: dict[str, str]) -> ast.AST | None:
    """The key expression of a read, direct or through a resolver, or None."""
    key = _direct_key(node)
    if key is not None:
        return key
    if isinstance(node, ast.Call):
        name = _callee(node)
        index = resolvers.get(aliases.get(name or "", name or ""))
        if index is not None and len(node.args) > index:
            return node.args[index]
    return None


def environment_resolvers(trees: dict[str, ast.AST]) -> dict[str, int]:
    """``{function name: index of the parameter it reads the environment through}``.

    A function is a resolver when its body reads the environment - directly, or
    through a resolver already found - with a key that is one of its own
    positional parameters. Iterated to a fixed point so ``hz_from_env`` is found
    even when it only delegates to ``_float_env``.
    """
    functions = [
        (node, _import_aliases(tree))
        for tree in trees.values()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    resolvers: dict[str, int] = {}
    grown = True
    while grown:
        grown = False
        for function, aliases in functions:
            if function.name in resolvers:
                continue
            parameters = [arg.arg for arg in function.args.posonlyargs + function.args.args]
            for node in ast.walk(function):
                key = _read_key(node, resolvers, aliases)
                if isinstance(key, ast.Name) and key.id in parameters:
                    resolvers[function.name] = parameters.index(key.id)
                    grown = True
                    break
    return resolvers


def names_read(trees: dict[str, ast.AST]) -> dict[str, list[str]]:
    """``{name: [label:line, ...]}`` for every own-prefix key the trees read."""
    resolvers = environment_resolvers(trees)
    found: dict[str, list[str]] = {}
    for label, tree in trees.items():
        aliases = _import_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Call, ast.Subscript)):
                continue
            key = _read_key(node, resolvers, aliases)
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.startswith(OWN_PREFIX):
                found.setdefault(key.value, []).append(f"{label}:{node.lineno}")
    return found


def _parse(sources: dict[str, str]) -> dict[str, ast.AST]:
    return {label: ast.parse(source, filename=label) for label, source in sources.items()}


def package_trees() -> dict[str, ast.AST]:
    return _parse(
        {
            str(module.relative_to(REPO_ROOT)): module.read_text(encoding="utf-8")
            for module in sorted(PACKAGE.rglob("*.py"))
        }
    )


def documented_names(pages: dict[str, str]) -> set[str]:
    """Every name the pages document, in full or by sibling shorthand.

    A shorthand ``_SUFFIX`` documents ``PREFIX_SUFFIX`` only when the same page
    also spells some full name ``PREFIX_...`` out: the prefix is read off the
    documented sibling, so the suffix alone proves nothing.
    """
    names: set[str] = set()
    for text in pages.values():
        full = set(_FULL_NAME.findall(text))
        names |= full
        prefixes = {name[:idx] for name in full for idx, char in enumerate(name) if char == "_"}
        for suffix in _SHORTHAND.findall(text):
            names |= {prefix + suffix for prefix in prefixes if (prefix + suffix).startswith(OWN_PREFIX)}
    return names


def _load_pages() -> dict[str, str]:
    return {str(page.relative_to(REPO_ROOT)): page.read_text(encoding="utf-8") for page in PAGES}


def test_every_environment_variable_the_package_reads_is_documented() -> None:
    read = names_read(package_trees())
    pages = _load_pages()
    documented = documented_names(pages)

    assert len(read) >= MINIMUM_NAMES_READ, (
        f"the walk over {PACKAGE} found {len(read)} {OWN_PREFIX}* names, below the floor of "
        f"{MINIMUM_NAMES_READ}; the read shapes this test recognises have drifted from the package"
    )
    naming_pages = [page for page, text in pages.items() if _FULL_NAME.search(text)]
    assert len(naming_pages) >= MINIMUM_PAGES_NAMING_ONE, (
        f"only {len(naming_pages)} page(s) name a {OWN_PREFIX}* variable; the reference pages have moved"
    )

    undocumented = sorted(name for name in read if name not in documented)
    assert not undocumented, (
        f"{len(undocumented)} environment variable(s) the package reads appear in no page under "
        f"README.md or docs/:\n"
        + "\n".join(f"  {name}  read at {', '.join(read[name])}" for name in undocumented)
        + "\nAdd a row to the README's 'Environment variables' table (or the docs page that owns "
        "the subsystem) naming the variable, what it selects, and its default."
    )


class TestTheReadShapesAreAllRecognised:
    """The population is only as complete as the shapes the walk recognises."""

    @pytest.mark.parametrize(
        "source",
        [
            'import os\nx = os.getenv("STRANDS_PROBE")\n',
            'import os\nx = os.getenv("STRANDS_PROBE", "default")\n',
            'import os\nx = os.environ.get("STRANDS_PROBE")\n',
            'import os\nx = os.environ.setdefault("STRANDS_PROBE", "1")\n',
            'import os\nx = os.environ["STRANDS_PROBE"]\n',
            'from os import environ\nx = environ.get("STRANDS_PROBE")\n',
            'from os import getenv\nx = getenv("STRANDS_PROBE")\n',
        ],
    )
    def test_a_read_is_seen_however_it_is_spelled(self, source: str) -> None:
        assert list(names_read(_parse({"probe.py": source}))) == ["STRANDS_PROBE"]

    def test_a_read_through_a_resolver_is_seen(self) -> None:
        source = (
            "import os\n"
            "def _int_env(name, default):\n"
            '    return int(os.getenv(name, "") or default)\n'
            'CAP = _int_env("STRANDS_PROBE", 4)\n'
        )
        assert names_read(_parse({"probe.py": source})) == {"STRANDS_PROBE": ["probe.py:4"]}

    def test_a_resolver_that_delegates_to_another_is_seen(self) -> None:
        source = (
            "import os\n"
            "def _float_env(name, default):\n"
            '    return float(os.getenv(name, "") or default)\n'
            "def hz_from_env(default, name):\n"
            "    return _float_env(name, default)\n"
            'HZ = hz_from_env(10.0, "STRANDS_PROBE")\n'
        )
        assert names_read(_parse({"probe.py": source})) == {"STRANDS_PROBE": ["probe.py:6"]}

    def test_a_resolver_imported_under_an_alias_is_seen(self) -> None:
        trees = _parse(
            {
                "config.py": 'import os\ndef _bool_env(name, default=False):\n    return os.getenv(name, "") == "1"\n',
                "core.py": 'from .config import _bool_env as _zc_bool_env\nON = _zc_bool_env("STRANDS_PROBE")\n',
            }
        )
        assert names_read(trees) == {"STRANDS_PROBE": ["core.py:2"]}

    def test_a_function_that_only_names_the_variable_in_a_message_is_not_a_resolver(self) -> None:
        """Passing the name to a refusal's wording is not a read of it."""
        source = (
            "def _refuse(name, raw):\n"
            '    raise ValueError(f"{name}={raw!r} is unusable")\n'
            '_refuse("STRANDS_PROBE", "x")\n'
        )
        assert names_read(_parse({"probe.py": source})) == {}

    def test_a_name_inside_a_string_literal_is_not_a_read(self) -> None:
        """Shipped Lambda source is text to this process, not a read it makes."""
        source = 'BODY = """\nimport os\n_TABLE = os.environ.get("STRANDS_PROBE")\n"""\n'
        assert names_read(_parse({"probe.py": source})) == {}

    def test_a_name_outside_the_owned_prefix_is_not_graded(self) -> None:
        assert names_read(_parse({"probe.py": 'import os\nx = os.getenv("MUJOCO_GL")\n'})) == {}


class TestAPageDocumentsANameOnlyByNamingIt:
    """The documented set errs towards refusing, so a gap cannot hide in a match."""

    def test_a_longer_name_does_not_document_its_prefix(self) -> None:
        assert documented_names({"p.md": "`STRANDS_PROBE_ALLOW` widens the allowlist"}) == {"STRANDS_PROBE_ALLOW"}

    def test_a_sibling_shorthand_documents_the_name_it_abbreviates(self) -> None:
        page = "| `STRANDS_MESH_POSE_HZ`, `_IMU_HZ` | per-topic rate |"
        assert documented_names({"p.md": page}) >= {"STRANDS_MESH_POSE_HZ", "STRANDS_MESH_IMU_HZ"}

    def test_a_shorthand_with_no_documented_sibling_documents_nothing(self) -> None:
        assert documented_names({"p.md": "set `_IMU_HZ` to 0"}) == set()

    def test_a_shorthand_is_read_against_its_own_page(self) -> None:
        pages = {"a.md": "`STRANDS_MESH_POSE_HZ`", "b.md": "`_IMU_HZ`"}
        assert "STRANDS_MESH_IMU_HZ" not in documented_names(pages)
