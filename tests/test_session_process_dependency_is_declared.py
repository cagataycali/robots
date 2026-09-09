"""The session tools' process dependency is declared, and no hard import is not.

``lerobot_train`` and ``lerobot_teleoperate`` each launch a detached background
process and then answer ``status`` / ``stop`` / ``list`` about it. Both reach for
``psutil`` to do that -- directly, and again through
``strands_robots.tools._process_stop``, which owns the two questions a ``stop``
verb cannot answer without it: did the signalled process actually exit, and is
this pid still the process the session record was written for. All three imports
are unconditional, at module scope, so ``psutil`` is a requirement of *importing*
either tool, not of some branch inside it.

Nothing declared it. Measured against the checked-in lock, ``pip install
'strands-robots[lerobot]'`` resolved to 129 packages with no ``psutil`` among
them -- and that command is the remedy ``lerobot_teleoperate`` itself prints for
its own absent dependency, so following it produced an environment where the tool
raised ``ModuleNotFoundError: No module named 'psutil'`` before it could reach
that message. Two properties of the extras table let it survive:

* ``psutil`` reached CI only as a transitive of ``accelerate``, under the GPU
  extras (``[molmoact2]``, ``[smolvla]``, ``[kimodo]``, ``[cosmos3-diffusers]``).
  CI installs ``[all,dev]``, which folds three of those in, so the suite was
  green over an extra that could not import the tools it exists to serve. The
  parametrized cell below states that per extra: ``[lerobot]`` and
  ``[lerobot-async]`` are the two rows that were red, and the three that passed
  passed for a reason unrelated to the session tools.
* Nothing graded the *shape*. ``pip`` reports success on an extra that installs
  none of what an import needs, and ``ruff``/``mypy`` cannot know that a
  module-scope ``import psutil`` implies a packaging obligation.

So the rule pinned here is the general one rather than the psutil row alone:
**every third-party module this package imports unconditionally at module scope
is either named by a declared requirement, or recorded as a transitive with the
requirement that supplies it.** A new hard import satisfies neither and fails
``test_every_unconditional_import_is_named_or_a_recorded_transitive``, which is
the only thing in the tree that refuses that shape.

"Reachable in some install" would have been the easier rule and it is too weak:
``psutil`` *was* reachable, in ``[all]``, through ``accelerate`` -- so a rule
asking only for reachability passes on the very defect this change fixes. The
distinction between naming a dependency and inheriting one is therefore the
rule, and the transitive roster is where inheriting has to be argued: one entry
today, ``pyserial``, which arrives through ``lerobot[feetech]``'s own servo SDK,
where naming it separately would claim a dependency this package does not have.

Each recorded transitive is then verified against ``uv.lock`` -- uv's own
resolution of the manifest, and the artifact
``tests/test_lockfile_parity_gate.py`` already treats as authoritative -- so the
claim is checked rather than asserted, offline. Markers are not evaluated when
walking that graph, which makes reachability a superset: the walk can call a
distribution supplied when a marker would exclude it on some platform. That
direction favours not failing a correct manifest, and the naming half of the
rule is exact, so nothing rests on the superset alone.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

import strands_robots
from tests.uv_lock_closure import lock_closure

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

#: The shipped package, reached through the imported module so a layout change
#: cannot silently narrow the scan to nothing.
_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent

#: Modules whose session verbs cannot be imported without ``psutil``.
_PSUTIL_IMPORTERS = (
    "tools/_process_stop.py",
    "tools/lerobot_train.py",
    "tools/lerobot_teleoperate.py",
)

#: Extras that ship the two session tools' dependency stack: ``[lerobot]`` and
#: everything that folds it in. Each must supply ``psutil``, and three of them
#: did so already by accident -- via ``accelerate`` under a VLA's aux deps --
#: which is the accident this pin removes reliance on.
_EXTRAS_REACHING_THE_SESSION_TOOLS = ("lerobot", "lerobot-async", "molmoact2", "smolvla", "all")

#: Import name -> distribution that supplies it, for every third-party module
#: this package imports unconditionally at module scope. Only the rows where the
#: two names differ need explaining: ``serial`` is ``pyserial``, ``jwt`` is
#: ``PyJWT``, ``strands`` is ``strands-agents``, and ``device_connect_edge``
#: keeps its dashes. Names are compared canonically (PEP 503), so the manifest's
#: ``PyJWT`` and this table's ``pyjwt`` are the same distribution. A new hard
#: import must add its row here, which is the point: the addition is where a
#: packaging obligation gets noticed.
_DISTRIBUTION_BY_IMPORT_NAME = {
    "device_connect_edge": "device-connect-edge",
    "fastapi": "fastapi",
    "jwt": "pyjwt",
    "msgpack": "msgpack",
    "numpy": "numpy",
    "psutil": "psutil",
    "serial": "pyserial",
    "strands": "strands-agents",
    "torch": "torch",
    "webauthn": "webauthn",
}

#: Distributions above that no declared requirement names, with the requirement
#: that supplies each. An entry is an argument that inheriting is right here, not
#: an exemption: ``pyserial`` is the wire library of the servo SDK
#: ``lerobot[feetech]`` pulls, so its version is that SDK's to choose, and naming
#: it separately would claim a direct dependency on a transport this package
#: reaches only through the bus it drives. Verified reachable below, so the claim
#: is checked rather than trusted.
_SUPPLIED_TRANSITIVELY = {
    "pyserial": "lerobot[feetech] -> feetech-servo-sdk",
}


def _manifest() -> dict:
    with _PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def _requirements(extra: str | None) -> list[Requirement]:
    """Requirement objects the manifest declares for *extra* (``None`` = base)."""
    project = _manifest()
    declared = project["dependencies"] if extra is None else project["optional-dependencies"][extra]
    return [Requirement(text) for text in declared]


def _every_closure() -> dict[str | None, frozenset[str]]:
    extras: list[str | None] = [None]
    extras += sorted(_manifest()["optional-dependencies"])
    return {extra: lock_closure(extra) for extra in extras}


def _module_scope_imports(source: Path) -> set[str]:
    """Top-level names *source* imports unconditionally.

    Only ``tree.body`` is walked, so an import inside ``try:`` (the shape an
    optional dependency with a fallback takes) or under ``if TYPE_CHECKING:`` is
    not one of these: neither makes the module unimportable when absent.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _unconditional_third_party_imports() -> dict[str, list[str]]:
    """Third-party import name -> the modules that hard-import it at module scope."""
    found: dict[str, list[str]] = {}
    for source in sorted(_PACKAGE_ROOT.rglob("*.py")):
        for name in _module_scope_imports(source):
            if name in sys.stdlib_module_names or name in {"__future__", "strands_robots"}:
                continue
            # A sibling module ships in the same directory: first-party by
            # layout, not a distribution. `policies/vera/docker/launch_server.py`
            # imports its own `wandb_offline_resolve` this way.
            if (source.parent / f"{name}.py").exists():
                continue
            found.setdefault(name, []).append(str(source.relative_to(_PACKAGE_ROOT)))
    return found


def test_the_lerobot_extra_declares_the_process_dependency() -> None:
    """``[lerobot]`` names psutil, so the tools' own install hint supplies it."""
    named = {str(canonicalize_name(requirement.name)) for requirement in _requirements("lerobot")}
    assert "psutil" in named, (
        "the lerobot extra does not declare psutil, so `pip install "
        "'strands-robots[lerobot]'` -- what lerobot_teleoperate prints as the "
        "remedy for its own absent dependency -- installs an environment where "
        f"importing either session tool raises ModuleNotFoundError. Declared: {sorted(named)}"
    )


@pytest.mark.parametrize("extra", _EXTRAS_REACHING_THE_SESSION_TOOLS)
def test_every_extra_that_ships_the_session_tools_locks_psutil(extra: str) -> None:
    """Each extra carrying the session tools' stack resolves to a psutil."""
    closure = lock_closure(extra)
    assert "psutil" in closure, (
        f"[{extra}] locks {len(closure)} distributions and psutil is not among "
        "them, so an install of that extra ships the session tools and none of "
        "the process-lifecycle module they import at module scope"
    )


@pytest.mark.parametrize("module", _PSUTIL_IMPORTERS)
def test_the_session_tools_import_psutil_unconditionally(module: str) -> None:
    """psutil is a requirement of importing these, which is why declaring is the fix.

    A gated import with a fallback would be a different remedy; these are not
    that shape, and this cell is the premise the packaging cells rest on. It
    fails if a module stops hard-importing psutil, at which point the obligation
    -- not just its remedy -- has changed.
    """
    assert "psutil" in _module_scope_imports(_PACKAGE_ROOT / module), (
        f"{module} no longer imports psutil at module scope. If the dependency is "
        "now optional there, this file's premise is stale: re-decide whether the "
        "extra should still declare it before deleting this cell"
    )


def _named_distributions() -> set[str]:
    """Every distribution the manifest names, in base or in any extra.

    Self-referencing entries (``strands-robots[lerobot]``) are folded out: they
    name this project, not a dependency, and their contents are read from the
    extra they point at.
    """
    project = _manifest()
    groups = [project["dependencies"], *project["optional-dependencies"].values()]
    names = {str(canonicalize_name(Requirement(text).name)) for group in groups for text in group}
    return names - {str(canonicalize_name("strands-robots"))}


def test_every_unconditional_import_is_named_or_a_recorded_transitive() -> None:
    """A hard module-scope import is declared, or its supplier is written down.

    The general rule, and the reason this file is not three psutil assertions.
    Reachability alone would not have caught psutil -- ``[all]`` reached it
    through ``accelerate`` -- so what is graded is whether the manifest takes
    responsibility for the dependency: by naming it, or by recording which
    requirement it is inherited from and why.
    """
    found = _unconditional_third_party_imports()

    # Non-vacuity: an empty scan would agree with everything.
    assert len(found) >= 8, f"the import scan found only {sorted(found)}, so it is not reading the package"
    assert "psutil" in found, "psutil is no longer hard-imported anywhere; this file's subject has moved"

    unrostered = sorted(set(found) - set(_DISTRIBUTION_BY_IMPORT_NAME))
    assert not unrostered, (
        f"{unrostered} are imported at module scope with no distribution named for "
        "them. Add each to _DISTRIBUTION_BY_IMPORT_NAME, then declare it or record "
        f"what supplies it. Sites: { {name: found[name] for name in unrostered} }"
    )

    named = _named_distributions()
    unaccounted = {
        _DISTRIBUTION_BY_IMPORT_NAME[import_name]: sites
        for import_name, sites in sorted(found.items())
        if _DISTRIBUTION_BY_IMPORT_NAME[import_name] not in named
        and _DISTRIBUTION_BY_IMPORT_NAME[import_name] not in _SUPPLIED_TRANSITIVELY
    }
    assert not unaccounted, (
        "these are imported at module scope and the manifest neither names them "
        "nor records what supplies them, so an install can ship the module and "
        f"none of its imports: {unaccounted}. Declare each in the extra whose "
        "capability needs it, or add it to _SUPPLIED_TRANSITIVELY with the "
        "requirement it arrives through"
    )


@pytest.mark.parametrize("distribution", sorted(_SUPPLIED_TRANSITIVELY))
def test_a_recorded_transitive_really_is_supplied(distribution: str) -> None:
    """The requirement a transitive is claimed to arrive through does supply it.

    Without this, ``_SUPPLIED_TRANSITIVELY`` would be a way to silence the rule
    rather than a claim about the resolution.
    """
    suppliers = [extra for extra, closure in _every_closure().items() if distribution in closure]
    assert suppliers, (
        f"{distribution} is recorded as arriving through "
        f"{_SUPPLIED_TRANSITIVELY[distribution]!r}, and no declared extra resolves "
        "to it. Either the requirement stopped supplying it -- in which case the "
        "import needs a declaration -- or the note is wrong"
    )


def test_the_declared_floor_ships_the_api_the_session_verbs_call() -> None:
    """The installed psutil satisfies the declared range and carries what is called.

    The attribute list is read out of the three modules rather than restated, so
    a new psutil API reached for is graded against the floor rather than assumed
    to predate it.
    """
    psutil = pytest.importorskip("psutil")

    declared = [
        requirement for requirement in _requirements("lerobot") if canonicalize_name(requirement.name) == "psutil"
    ]
    assert declared, "the lerobot extra declares no psutil, so there is no floor to grade"
    specifiers = SpecifierSet(str(declared[0].specifier))
    assert Version(psutil.__version__) in specifiers, (
        f"the installed psutil {psutil.__version__} is outside the declared "
        f"{specifiers}, so this cell grades a version no install resolves"
    )

    used: set[str] = set()
    for module in _PSUTIL_IMPORTERS:
        tree = ast.parse((_PACKAGE_ROOT / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "psutil":
                used.add(node.attr)
    assert len(used) >= 5, f"the attribute scan found only {sorted(used)}, so it is not reading the modules"

    absent = sorted(name for name in used if not hasattr(psutil, name))
    assert not absent, (
        f"psutil {psutil.__version__} is inside the declared floor and does not "
        f"carry {absent}, which the session verbs call. The floor is too low"
    )
