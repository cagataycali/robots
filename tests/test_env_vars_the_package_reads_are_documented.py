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
  documented in ``docs/security/policy-code.md`` with no mention of the variable it
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
not one.

A key need not be a literal at the read site either. A module can bind the
name once and read through the binding - ``RDZV_TIMEOUT_ENV =
"STRANDS_TRAIN_RDZV_TIMEOUT_S"`` and then ``os.environ.get(RDZV_TIMEOUT_ENV)``
- which is the shape a module reaches for when the same name is also spelled
into a refusal or a log line. The walk follows a ``Name`` key to a module-scope
string constant, and a receiver that selects the environment conditionally
(``(env if env is not None else os.environ).get(KEY)``, the injectable-mapping
idiom) counts as ``os.environ``. Measured on the tree this arrived in, three
names reached the environment only that way and appeared in no page:
``STRANDS_TRAIN_EXTRA_FLAGS_ALLOW`` - the allowlist a headless ``lerobot_train``
refusal names as its own remedy - and ``STRANDS_TRAIN_RDZV_TIMEOUT_S`` /
``STRANDS_TRAIN_LOCAL_ADDR``, the two bounds on an elastic launch's rendezvous.
A ``Name`` bound anywhere else (a parameter, a local) still names nothing a
page could spell and is not graded. A page is any of ``README.md`` and
``docs/**/*.md``: ``docs/security/mesh.md`` already owns the AWS IoT credentials
and the mesh TLS material, graded by their own reference tests, and this test
does not move them. It also honours the README's shorthand for a family of
sibling names (```STRANDS_MESH_POSE_HZ`, `_IMU_HZ`, ...``) - a suffix counts
only when a documented full name shares its prefix, so a bare suffix with no
sibling documents nothing.

A key may also be *built* at the read site rather than named there. A module
that owns a family of variables binds the family's prefix once and reads each
member as ``os.getenv(_ENV + "ENABLED")`` - or, for the members it validates,
through a resolver whose body spells ``var = _ENV + name`` and reads ``var``.
Neither shape puts a whole name anywhere the walk above could see it: the
constant holds a prefix, the literal holds a suffix, and the resolver's key is
a local rather than one of its parameters. Measured on the tree this arrived
in, ``dashboard/auth.py`` read twelve ``STRANDS_DASH_AUTH_*`` names that way
and no page named one of them - among them the ``ORIGIN`` and ``RP_ID`` the
module's own refusals tell an operator to set, and ``TOKEN_TTL``, whose
refusal exists so a narrowed session window is never silently widened. The
walk concatenates a ``+`` chain of literals and module constants, and a
resolver may prepend or append such text to the parameter it reads through;
the parameter must appear exactly once, and a chain that involves a local or
a second parameter still names nothing a page could spell.

Out of scope, and why: a name that appears only inside a string literal is not
read by this process. ``mesh.iot.bootstrap`` ships the e-stop fan-out Lambda's
source as text and sets that Lambda's ``STRANDS_SAFETY_TABLE`` itself, so the
variable is provisioned rather than exposed, and the AST walk does not see it
by construction.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple

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
    """Is ``node`` the environment mapping - outright, or on one arm of a conditional?

    ``(env if env is not None else os.environ)`` is how a function takes an
    injectable mapping and still reads the real environment by default, and the
    read is a read of the environment on the arm a caller does not override.
    """
    if isinstance(node, ast.IfExp):
        return _is_environ(node.body) or _is_environ(node.orelse)
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


class Resolver(NamedTuple):
    """How a function reads the environment through one of its parameters.

    ``parameter`` is the index of the positional parameter the key is built from; ``prefix`` and
    ``suffix`` are the literal text the function puts around it, empty for the
    common resolver that reads its parameter as the whole key.
    """

    parameter: int
    prefix: str
    suffix: str


def _concatenation(*parts: ast.expr) -> ast.expr:
    """The ``+`` chain of *parts*, as the key expression a resolver call reads."""
    expression = parts[0]
    for part in parts[1:]:
        expression = ast.BinOp(left=expression, op=ast.Add(), right=part)
    return expression


def _read_key(node: ast.AST, resolvers: dict[str, Resolver], aliases: dict[str, str]) -> ast.AST | None:
    """The key expression of a read, direct or through a resolver, or None.

    A resolver that wraps its parameter in literal text reads a key the call
    site never spells whole, so the returned expression is the concatenation
    the resolver performs, with the call's own argument in the middle.
    """
    key = _direct_key(node)
    if key is not None:
        return key
    if isinstance(node, ast.Call):
        name = _callee(node)
        resolver = resolvers.get(aliases.get(name or "", name or ""))
        if resolver is not None and len(node.args) > resolver.parameter:
            argument = node.args[resolver.parameter]
            if not (resolver.prefix or resolver.suffix):
                return argument
            return _concatenation(ast.Constant(resolver.prefix), argument, ast.Constant(resolver.suffix))
    return None


def _addends(node: ast.AST) -> list[ast.AST]:
    """The operands of a ``+`` chain, left to right; a non-chain is its own single operand."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _addends(node.left) + _addends(node.right)
    return [node]


def _locals_bound_once(function: ast.AST) -> dict[str, ast.AST]:
    """``{name: value}`` for every local *function* assigns exactly once to a bare name.

    A resolver spells ``var = _ENV + name`` and then reads ``var``; following
    that one binding is what lets the key be read back to the parameter. A
    name bound twice is ambiguous and is left unresolved.
    """
    seen: dict[str, list[ast.AST]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            seen.setdefault(node.targets[0].id, []).append(node.value)
    return {name: values[0] for name, values in seen.items() if len(values) == 1}


def _parameter_template(key: ast.AST, parameters: list[str], constants: dict[str, str]) -> Resolver | None:
    """Read *key* as literal text around exactly one of *parameters*, or None.

    Each operand of the ``+`` chain must be a string literal, a module string
    constant, or a parameter; the parameter must occur once. Anything else -
    a local, a call, two parameters - is not a key the caller's argument
    determines, so the function is not a resolver through it.
    """
    parameter: int | None = None
    prefix, suffix = "", ""
    for operand in _addends(key):
        if isinstance(operand, ast.Name) and operand.id in parameters:
            if parameter is not None:
                return None
            parameter = parameters.index(operand.id)
            continue
        text = _key_string(operand, constants)
        if text is None:
            return None
        if parameter is None:
            prefix += text
        else:
            suffix += text
    if parameter is None:
        return None
    return Resolver(parameter, prefix, suffix)


def environment_resolvers(trees: dict[str, ast.AST]) -> dict[str, Resolver]:
    """``{function name: how it reads the environment through a parameter}``.

    A function is a resolver when its body reads the environment - directly, or
    through a resolver already found - with a key built from one of its own
    positional parameters: the parameter itself, or literal text around it,
    possibly through a local bound once (``var = _ENV + name``). Iterated to a
    fixed point so ``hz_from_env`` is found even when it only delegates to
    ``_float_env``.
    """
    # A module's aliases and constants, and a function's once-bound locals, do
    # not depend on which resolvers are known yet, so each is read once here:
    # per module for the first two, per function for the third. Reading the
    # aliases beside each function instead walked the whole module once for
    # every function it defines - 4,684 module walks over 311 files - and that
    # was 84% of this cell's time under a profile; 26 s -> 6 s for the file.
    functions: list[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef, dict[str, str], dict[str, str], dict[str, ast.AST]]
    ] = []
    for tree in trees.values():
        aliases = _import_aliases(tree)
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append((node, aliases, constants, _locals_bound_once(node)))
    resolvers: dict[str, Resolver] = {}
    grown = True
    while grown:
        grown = False
        for function, aliases, constants, bound_once in functions:
            if function.name in resolvers:
                continue
            parameters = [arg.arg for arg in function.args.posonlyargs + function.args.args]
            for node in ast.walk(function):
                key = _read_key(node, resolvers, aliases)
                if key is None:
                    continue
                if isinstance(key, ast.Name) and key.id in bound_once and key.id not in parameters:
                    key = bound_once[key.id]
                resolver = _parameter_template(key, parameters, constants)
                if resolver is not None:
                    resolvers[function.name] = resolver
                    grown = True
                    break
    return resolvers


def _module_string_constants(tree: ast.AST) -> dict[str, str]:
    """``{name: value}`` for every module-scope ``NAME = "literal"`` in ``tree``.

    Only the module body is read, so a parameter or a local of the same name
    inside a function is not mistaken for the constant. A name bound twice at
    module scope keeps its last binding, which is what the reads after it see.
    """
    constants: dict[str, str] = {}
    body = getattr(tree, "body", [])
    for statement in body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value.value
    return constants


def _key_string(key: ast.AST, constants: dict[str, str]) -> str | None:
    """The string a key expression names, or None.

    A literal, a module constant bound to one, or a ``+`` chain of those -
    ``_ENV + "ENABLED"`` names ``STRANDS_DASH_AUTH_ENABLED`` when ``_ENV`` is
    bound to the prefix at module scope. A chain with an operand that names no
    string names nothing.
    """
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    if isinstance(key, ast.Name):
        return constants.get(key.id)
    if isinstance(key, ast.BinOp) and isinstance(key.op, ast.Add):
        left = _key_string(key.left, constants)
        right = _key_string(key.right, constants)
        if left is None or right is None:
            return None
        return left + right
    return None


def names_read(trees: dict[str, ast.AST]) -> dict[str, list[str]]:
    """``{name: [label:line, ...]}`` for every own-prefix key the trees read."""
    resolvers = environment_resolvers(trees)
    found: dict[str, list[str]] = {}
    for label, tree in trees.items():
        aliases = _import_aliases(tree)
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Call, ast.Subscript)):
                continue
            key = _read_key(node, resolvers, aliases)
            if key is None:
                continue
            name = _key_string(key, constants)
            if name is not None and name.startswith(OWN_PREFIX):
                found.setdefault(name, []).append(f"{label}:{node.lineno}")
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

    def test_a_read_through_a_module_constant_is_seen(self) -> None:
        """The name is bound once and read through the binding, as ``_inproc`` does."""
        source = 'import os\nKEY_ENV = "STRANDS_PROBE"\nx = os.environ.get(KEY_ENV, "")\n'
        assert names_read(_parse({"probe.py": source})) == {"STRANDS_PROBE": ["probe.py:3"]}

    def test_a_module_constant_handed_to_a_resolver_is_seen(self) -> None:
        source = (
            "import os\n"
            'KEY_ENV = "STRANDS_PROBE"\n'
            "def _int_env(name, default):\n"
            '    return int(os.getenv(name, "") or default)\n'
            "CAP = _int_env(KEY_ENV, 4)\n"
        )
        assert names_read(_parse({"probe.py": source})) == {"STRANDS_PROBE": ["probe.py:5"]}

    def test_a_read_off_a_conditional_environment_receiver_is_seen(self) -> None:
        """``(env if env is not None else os.environ).get(...)`` - the injectable-mapping idiom."""
        source = (
            "import os\n"
            "def read(env=None):\n"
            '    return (env if env is not None else os.environ).get("STRANDS_PROBE", "")\n'
        )
        assert names_read(_parse({"probe.py": source})) == {"STRANDS_PROBE": ["probe.py:3"]}

    def test_a_key_concatenated_onto_a_module_prefix_is_seen(self) -> None:
        """``os.getenv(_ENV + "ENABLED")`` - the family-prefix idiom ``dashboard.auth`` uses."""
        source = 'import os\n_ENV = "STRANDS_PROBE_"\nx = os.getenv(_ENV + "ON", "")\n'
        assert names_read(_parse({"probe.py": source})) == {"STRANDS_PROBE_ON": ["probe.py:3"]}

    def test_a_resolver_that_wraps_its_parameter_in_a_prefix_is_seen(self) -> None:
        """``var = _ENV + name`` then ``os.getenv(var)``: the key is a local, built from the parameter."""
        source = (
            "import os\n"
            '_ENV = "STRANDS_PROBE_"\n'
            "def _duration(name):\n"
            "    var = _ENV + name\n"
            '    return int(os.getenv(var, "") or 0)\n'
            'TTL = _duration("TTL")\n'
        )
        assert names_read(_parse({"probe.py": source})) == {"STRANDS_PROBE_TTL": ["probe.py:6"]}

    def test_a_resolver_that_appends_to_its_parameter_is_seen(self) -> None:
        source = (
            "import os\n"
            "def _hz(topic):\n"
            '    return float(os.getenv("STRANDS_PROBE_" + topic + "_HZ", "") or 1.0)\n'
            'HZ = _hz("POSE")\n'
        )
        assert names_read(_parse({"probe.py": source})) == {"STRANDS_PROBE_POSE_HZ": ["probe.py:4"]}

    def test_a_key_built_from_two_parameters_names_nothing(self) -> None:
        """No single argument determines the key, so the caller's literal is not a name."""
        source = (
            'import os\ndef _read(prefix, name):\n    return os.getenv(prefix + name)\n_read("STRANDS_PROBE_", "ON")\n'
        )
        assert names_read(_parse({"probe.py": source})) == {}

    def test_a_prefix_bound_only_inside_the_resolver_names_nothing(self) -> None:
        """A local prefix is not a module constant, so the chain names nothing a page could spell."""
        source = (
            "import os\n"
            "def _read(name):\n"
            '    prefix = "STRANDS_PROBE_"\n'
            "    return os.getenv(prefix + name)\n"
            '_read("ON")\n'
        )
        assert names_read(_parse({"probe.py": source})) == {}

    def test_a_name_bound_only_inside_a_function_is_not_a_module_constant(self) -> None:
        """A local of the constant's shape names nothing a page could spell; it stays ungraded."""
        source = 'import os\ndef read():\n    key = "STRANDS_PROBE"\n    return os.getenv(key)\n'
        assert names_read(_parse({"probe.py": source})) == {}

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


def test_a_module_is_read_for_its_aliases_once_however_many_functions_it_defines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alias walk is per module, not per function.

    ``environment_resolvers`` used to read a module's import aliases beside
    each of its functions, so a module with N functions was walked N times
    before a single resolver was looked for: 4,684 walks over the package's
    311 files, 84% of the whole-package cell. The population it grades is
    unchanged by reading them once, so this holds the count rather than the
    time, which a loaded runner cannot hold.
    """
    trees = _parse(
        {
            "a.py": "import os\n" + "\n".join(f"def f{i}(name):\n    return os.getenv(name)\n" for i in range(5)),
            "b.py": "import os\n" + "\n".join(f"def g{i}(x):\n    return x\n" for i in range(7)),
        }
    )
    labels = {id(tree): label for label, tree in trees.items()}
    calls: list[str] = []
    real = _import_aliases

    def counting(tree: ast.AST) -> dict[str, str]:
        calls.append(labels[id(tree)])
        return real(tree)

    monkeypatch.setattr(sys.modules[__name__], "_import_aliases", counting)

    resolvers = environment_resolvers(trees)

    assert set(resolvers) == {f"f{i}" for i in range(5)}, resolvers
    assert sorted(calls) == ["a.py", "b.py"], f"aliases read {len(calls)} times for 2 modules: {calls}"
