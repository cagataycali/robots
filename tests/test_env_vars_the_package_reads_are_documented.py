"""Every ``STRANDS_*`` environment variable the package reads is documented.

The README's Configuration section calls itself the single source of truth for
these variables, and ``AGENTS.md`` asks that a new one be added there in the
same pull request that introduces it. Nothing graded that: the reference pages
were checked one variable set at a time, each by a test written for the change
that added it (``tests/mesh/test_docs_*_env_var_reference.py``), so a variable
introduced without such a test was undocumented by default and the omission
was silent in the reassuring direction - the code honoured it, the tests that
set it passed, and no page a reader could reach named it.

Measured on the tree this arrived in, the package read 59 distinct ``STRANDS_*``
names and seven appeared in no page at all:

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

The population is derived from the package by AST rather than listed here, so
a variable added later is graded on arrival. A page is any of ``README.md`` and
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
#: tree this arrived in read 59 names across 62 sites and documented them on
#: 6 pages; both floors sit well below that.
MINIMUM_NAMES_READ = 40
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


def _names_read_from_source(source: str, label: str) -> dict[str, list[str]]:
    """``{name: [label:line, ...]}`` for every own-prefix key *source* reads."""
    found: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source, filename=label)):
        if not isinstance(node, (ast.Call, ast.Subscript)):
            continue
        key = _environment_key(node)
        if key is not None and key.startswith(OWN_PREFIX):
            found.setdefault(key, []).append(f"{label}:{node.lineno}")
    return found


def names_read_by_the_package() -> dict[str, list[str]]:
    """Every ``STRANDS_*`` variable the package reads, with the sites that read it."""
    found: dict[str, list[str]] = {}
    for module in sorted(PACKAGE.rglob("*.py")):
        label = str(module.relative_to(REPO_ROOT))
        for name, sites in _names_read_from_source(module.read_text(encoding="utf-8"), label).items():
            found.setdefault(name, []).extend(sites)
    return found


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
    read = names_read_by_the_package()
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
        assert list(_names_read_from_source(source, "probe.py")) == ["STRANDS_PROBE"]

    def test_a_name_inside_a_string_literal_is_not_a_read(self) -> None:
        """Shipped Lambda source is text to this process, not a read it makes."""
        source = 'BODY = """\nimport os\n_TABLE = os.environ.get("STRANDS_PROBE")\n"""\n'
        assert _names_read_from_source(source, "probe.py") == {}

    def test_a_name_outside_the_owned_prefix_is_not_graded(self) -> None:
        assert _names_read_from_source('import os\nx = os.getenv("MUJOCO_GL")\n', "probe.py") == {}


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
